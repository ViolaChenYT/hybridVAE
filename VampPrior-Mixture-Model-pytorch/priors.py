import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions import kl_divergence as kl
from torch.distributions.wishart import Wishart
import torch.nn.functional as F
from torch.distributions import constraints, transform_to
from torch.distributions import Dirichlet

import math
from typing import Union, Tuple, Any

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def dirichlet_log_norm(alpha: torch.Tensor) -> torch.Tensor:
    """log B(alpha) = sum lgamma(alpha_i) - lgamma(sum alpha_i)"""
    return torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(torch.sum(alpha, dim=-1))

def wishart_log_prob_cholesky(
    L: torch.Tensor,            # (..., D, D) lower-tri Cholesky of SPD matrix X
    df: float,                  # degrees of freedom
    scale_tril: torch.Tensor    # (..., D, D) Cholesky factor of the Wishart "scale" V
) -> torch.Tensor:
    """
    Compute log p(X) for X = L @ L.T under Wishart(df, V),
    where V is given via its Cholesky factor `scale_tril` (i.e., covariance parameterization).

    This mirrors TFP's WishartTriL(..., scale_tril=..., input_output_cholesky=True).
    """
    # Ensure parameter tensors live with L
    scale_tril = scale_tril.to(dtype=L.dtype, device=L.device)

    # Build the SPD matrix X from its Cholesky factor
    X = L @ L.transpose(-1, -2)   # (..., D, D)

    # Wishart with scale specified via its Cholesky factor
    W = Wishart(df=df, scale_tril=scale_tril)  # broadcasts over batch dims

    # Return log probability; shape broadcasts to batch of X
    return W.log_prob(X)

class Prior(nn.Module):
    """
    Base prior module in PyTorch.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent variable z.
    num_clusters : int
        Number of mixture components (if using a mixture prior).
    """
    def __init__(self, *, latent_dim: int, num_clusters: int, **kwargs):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_clusters = int(num_clusters)

    @torch.no_grad()
    def inference_step(self, **kwargs):
        """
        Optional fast path for inference-time computations (e.g., caching
        prior parameters). Override in subclasses as needed.

        Returns
        -------
        dict
            Any useful tensors/metadata for inference; empty by default.
        """
        return {}

    def kl_divergence(self, qz_x: td.Distribution, **kwargs) -> torch.Tensor:
        """
        Compute KL(q(z|x) || p(z)).

        Parameters
        ----------
        qz_x : torch.distributions.Distribution
            The approximate posterior over z (e.g., an Independent(Normal(...), 1)).

        Returns
        -------
        torch.Tensor
            KL divergence per-sample (shape: [batch]).
        """
        raise NotImplementedError("Implement in a subclass.")


class StandardNormal(Prior):
    def __init__(self, *, latent_dim: int, **kwargs):
        super().__init__(latent_dim=latent_dim,
                         num_clusters=kwargs.get("num_clusters") or 1)
        # p(z) = N(0, I) with diagonal covariance
        self.register_buffer("loc", torch.zeros(latent_dim))
        self.register_buffer("scale", torch.ones(latent_dim))

    @property
    def name(self) -> str:
        return "StandardNormal"

    @property
    def pz(self) -> td.Distribution:
        # Independent over latent dims
        return td.Independent(td.Normal(self.loc, self.scale), 1)

    def kl_divergence(self, qz_x: td.Distribution, **kwargs) -> torch.Tensor:
        # KL(q(z|x) || p(z)); expects qz_x to have batch shape [B]
        return kl(qz_x, self.pz)


class ClusteringPrior(Prior):
    def __init__(
        self,
        *,
        latent_dim: int,
        num_clusters: int,
        inference: str,
        learning_rate: float,
        **kwargs
    ):
        super().__init__(latent_dim=latent_dim, num_clusters=num_clusters)

        # inference
        assert inference in {None, "None", "MLE", "MAP", "MAP-DP"}
        self.inference = "None" if inference in {None, "None"} else str(inference)
        if self.inference != "None":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # ----- Priors (hyperpriors) -----
        # prior_alpha ~ Gamma(1, 1)
        self.prior_alpha_concentration = torch.as_tensor(1.0)
        self.prior_alpha_rate = torch.as_tensor(1.0)

        # prior_mu ~ N(0, I)
        # we'll use this via its log_prob as needed
        self.register_buffer("prior_mu_loc", torch.zeros(latent_dim))
        self.register_buffer("prior_mu_scale_tril", torch.eye(latent_dim))

        # prior_L ~ WishartTriL with df=latent_dim+2 and given scale_tril
        df = latent_dim + 2
        scale = (num_clusters ** (1.0 / latent_dim)) / math.sqrt(latent_dim + 2.0)
        self.df = float(df)
        self.register_buffer("prior_L_scale_tril", torch.eye(latent_dim) * scale)

        # ----- Mixture parameters -----
        # alpha (DP) — positive via softplus(raw_alpha)
        train_dp = ("DP" in self.inference)
        #raw_alpha_init = torch.as_tensor(1.0).log().expm1()  # inverse of softplus(1.0) approx
        raw_alpha_init = torch.as_tensor(1.0) + torch.log(-torch.expm1(-torch.as_tensor(1.0)))
        self.alpha = nn.Parameter(raw_alpha_init.clone(), requires_grad=train_dp)

        # mixture logits
        self.pi_logits = nn.Parameter(torch.zeros(num_clusters))

        # component precisions L (triangular Cholesky factors), one per component
        #raw_L0 = torch.eye(latent_dim).unsqueeze(0).repeat(num_clusters, 1, 1)
        # make diagonals inverse softplus so that softplus(diag)=1 initially
        #raw_L0.diagonal(dim1=-2, dim2=-1).copy_((torch.ones(num_clusters, latent_dim) - 1e-6).log())  # approx inv sp
        #self.raw_L = nn.Parameter(raw_L0)  # unconstrained; we map to tri-L via softplus on diag
        self.to_chol = transform_to(constraints.lower_cholesky)
        raw_L0 = torch.eye(latent_dim).expand(num_clusters, latent_dim, latent_dim).clone()
        self.L = nn.Parameter(to_chol.inv(raw_L0))

        # ----- Empirical Bayes tracker (running mean of -loss like tf.metrics.Mean) -----
        self.register_buffer("prior_loss_sum", torch.zeros(1))
        self.register_buffer("prior_loss_count", torch.zeros(1))


    # ---------- mixture & helpers ----------
    def pz_c(self, **kwargs) -> td.Distribution:
        """
        Component distributions p(z|c): subclass should implement to match TF design.
        Expected to return a batch of K distributions over R^D, each as Independent(Normal(...), 1) or MVN.
        Here we raise NotImplementedError to mirror your original code.
        """
        raise NotImplementedError

    def pz(self, **kwargs) -> td.Distribution:
        """
        Mixture prior p(z) = sum_c pi_c p(z|c).
        """
        cat = td.Categorical(logits=self.pi_logits)
        comp = self.pz_c(**kwargs)
        return td.MixtureSameFamily(cat, comp)

    def kl_divergence(self, qz_x: td.Distribution, **kwargs) -> torch.Tensor:
        """
        Monte Carlo KL: KL(q || p) = -H(q) - E_q[log p]
        """
        return -qz_x.entropy() - self.pz(**kwargs).log_prob(qz_x.sample())

    def cluster_probabilities(self, *, samples: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given samples z ~ q(z|x): return q(c|z) ∝ p(z|c) π_c
        samples: (B, D) → returns (B, K)
        """
        z = samples.unsqueeze(1)  # (B, 1, D)
        return self.qc(z, **kwargs)

    def log_prior_prob_pi(self) -> torch.Tensor:
        """
        Dirichlet prior over softmax(pi_logits) with concentration alpha_i = 1/(K*alpha).
        log p(pi) = sum (alpha_i - 1) log pi_i - log B(alpha)
        """
        K = self.num_clusters
        alpha = torch.ones(K, device=self.pi_logits.device, dtype=self.pi_logits.dtype) / (K * self.alpha)
        log_pi = F.log_softmax(self.pi_logits, dim=-1)
        return torch.sum((alpha - 1.0) * log_pi, dim=-1) - dirichlet_log_norm(alpha)

    def log_prior_prob_mu(self, **kwargs) -> torch.Tensor:
        """
        Placeholder to match your interface.
        Should return per-component log prior for component means, e.g., Normal(0, I).
        Implement in subclass if component means are parameters there.
        """
        raise NotImplementedError

    def log_prob_z_c(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        log p(z | c) for each component c. Should return shape (B, K).
        Implement in subclass to match the component family parameterization.
        """
        raise NotImplementedError

    def qc(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        q(c | z) ∝ exp( log p(z|c) + log pi_c )
        z: (B, 1, D) or (B, D) -> returns (B, K)
        """
        if z.dim() == 2:
            z = z.unsqueeze(1)
        log_prob_z_c = self.log_prob_z_c(z, **kwargs)                 # (B, K)
        log_prob_c = F.log_softmax(self.pi_logits, dim=-1)            # (K,)
        return F.softmax(log_prob_z_c + log_prob_c, dim=-1)           # (B, K)

    # ---------- EM-like update ----------
    def inference_step(self, *, encoder: callable, x: torch.Tensor, **kwargs):
        """
        E-step: compute responsibilities q(c|z).
        M-step: update mixture params by maximizing expected complete-data log-likelihood
                plus MAP terms if requested.
        Assumes encoder(x, **kwargs) returns a torch.distributions.Distribution with .sample().
        """
        # sample z ~ q(z|x)
        z = encoder(x, **kwargs).sample()      # (B, D)
        z = z.unsqueeze(1)                     # (B, 1, D)
        n = torch.tensor(z.size(0), dtype=torch.float32, device=z.device)

        # E-step & loss
        # Using autograd (no explicit GradientTape)
        # loss = - E_qc [ log p(z|c) + log pi_c ] (+ priors)
        log_prob_z_c = self.log_prob_z_c(z, encoder=encoder, **kwargs)      # (B, K)
        log_prob_c   = F.log_softmax(self.pi_logits, dim=-1)                 # (K,)
        qc = F.softmax(log_prob_z_c + log_prob_c, dim=-1)                    # (B, K)

        loss = -(qc * (log_prob_z_c + log_prob_c)).sum(dim=-1).mean()        # scalar

        if self.inference == "MAP-DP":
            # prior on alpha (Gamma(1,1))
            # log p(alpha) = (a-1) log alpha - b alpha + const; here a=b=1 -> -alpha + const
            # Use distribution for clarity:
            prior_alpha = td.Gamma(concentration=self.prior_alpha_concentration.to(z.device),
                                   rate=self.prior_alpha_rate.to(z.device))
            loss = loss - (prior_alpha.log_prob(self.alpha)).mean() / n.clamp_min(1.0)

        if "MAP" in self.inference:
            # Dirichlet prior over pi
            loss = loss - self.log_prior_prob_pi() / n.clamp_min(1.0)
            # Prior over component means (delegate)
            loss = loss - (self.log_prior_prob_mu(encoder=encoder, **kwargs).sum()) / n.clamp_min(1.0)
            # Wishart prior over precision Cholesky L_k
            # Sum_k log p(L_k) with df and scale_tril
            L = self.L
            lp_L = wishart_log_prob_cholesky(L, df=self.df, scale_tril=self.prior_L_scale_tril.to(L.device))
            loss = loss - lp_L.sum() / n.clamp_min(1.0)

        # M-step: gradient update
        if self.inference != "None":
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Optional: guard against NaNs/Infs in grads
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e3)
            self.optimizer.step()

        # running mean tracker (mirror of tf.metrics.Mean on -loss)
        with torch.no_grad():
            val = (-loss.detach()).reshape(1)
            self.prior_loss_sum += val
            self.prior_loss_count += torch.ones_like(val)

        # debug assertions for NaN/Inf
        with torch.no_grad():
            for name, p in self.named_parameters():
                if not torch.isfinite(p).all():
                    raise FloatingPointError(f"Non-finite parameter detected: {name}")

    @property
    def prior_loss(self) -> torch.Tensor:
        """Return running mean of (-loss)."""
        denom = torch.clamp(self.prior_loss_count, min=1.0)
        return self.prior_loss_sum / denom


class GaussianMixture(ClusteringPrior):
    def __init__(self, *, latent_dim: int, num_clusters: int, **kwargs):
        super().__init__(latent_dim=latent_dim, num_clusters=num_clusters, **kwargs)
        # mu ~ learnable means for each cluster: shape (K, D)
        self.mu = nn.Parameter(torch.randn(num_clusters, latent_dim))

    @property
    def name(self) -> str:
        return "GaussianMixture"

    # --- prior over mu: N(0, I) per component (independent across clusters) ---
    def log_prior_prob_mu(self, **kwargs) -> torch.Tensor:
        return self.prior_mu.log_prob(self.mu)

    # --- component distributions p(z | c) ---
    def pz_c(self, **kwargs) -> td.Distribution:
        """
        Return a batch of K multivariate normals over R^D.
        We interpret self.L as Cholesky(precision) for each component:
          precision_k = L_k @ L_k^T
        Using precision_matrix avoids forming covariance & its Cholesky.
        """
        L = self.L                                   # (K, D, D), lower-tri, positive diag
        precision = L @ L.transpose(-1, -2)          # (K, D, D)
        return td.MultivariateNormal(
            loc=self.mu,                              # (K, D)
            precision_matrix=precision               # (K, D, D)
        )

    # --- log p(z | c) for a batch of z ---
    def log_prob_z_c(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        z: (B, 1, D) or (B, D)
        returns: (B, K) of log p(z_b | c)
        """
        if z.dim() == 2:
            z = z.unsqueeze(1)                       # (B, 1, D)
        return self.pz_c(**kwargs).log_prob(z)       # broadcast → (B, K)


class VampPriorMixture(ClusteringPrior):
    def __init__(
        self,
        *,
        latent_dim: int,
        num_clusters: int,
        u: Union[Any, Tuple[Any, ...]],
        **kwargs
    ):
        super().__init__(latent_dim=latent_dim, num_clusters=num_clusters, **kwargs)
        self.u = u  # pseudo-inputs for VampPrior

        # Sanity checks (mirror your tf.assert_equal assumptions):
        # prior_mu is N(0, I) as registered in ClusteringPrior buffers
        with torch.no_grad():
            if not torch.allclose(self.prior_mu_loc, torch.zeros_like(self.prior_mu_loc)):
                raise ValueError("log_prior_prob_mu assumes prior_mu.loc == 0")
            I = torch.eye(self.latent_dim, device=self.prior_mu_scale_tril.device, dtype=self.prior_mu_scale_tril.dtype)
            cov = self.prior_mu_scale_tril @ self.prior_mu_scale_tril.T
            if not torch.allclose(cov, I):
                raise ValueError("log_prior_prob_mu assumes prior_mu.covariance == I")

    @property
    def name(self) -> str:
        return "VampPriorMixture"

    # ----- helper: query encoder on pseudo-inputs -----
    def qmu(self, encoder: callable, **kwargs) -> td.Distribution:
        """
        Return a multivariate normal distribution q(mu | u) for each pseudo-input.
        Expected batch shape: (K,), event shape: (D,).
        """
        return encoder(self.u, **kwargs)

    # ----- E_q[ log p(mu) ] with p(mu)=N(0, I) -----
    def log_prior_prob_mu(self, **kwargs) -> torch.Tensor:
        """
        Compute E_{q(mu)}[ log N(mu | 0, I) ] per component; returns (K,).
        Using: E[log N(x|0,I)] = -0.5 * (D*log(2π) + ||m||^2 + tr(S))
        """
        qmu: td.MultivariateNormal = self.qmu(**kwargs)  # assumes MVN
        m = qmu.mean                     # (K, D)
        S = qmu.covariance_matrix        # (K, D, D)
        D = m.shape[-1]

        # 0.5 * logdet(I / (2π)) = -0.5 * D * log(2π)
        const = -0.5 * D * math.log(2.0 * math.pi)

        # -0.5 * (||m||^2 + tr(S))
        sq_mahal = -0.5 * (m.pow(2).sum(dim=-1))
        tr_term  = -0.5 * torch.einsum("kii->k", S)     # trace per K

        expected_ll = const + sq_mahal + tr_term        # (K,)
        return expected_ll

    # ----- Components p(z | c) = N( mean = E_q[mu], cov = (precision^{-1}) + Cov_q[mu] ) -----
    def pz_c(self, **kwargs) -> td.Distribution:
        """
        For each component c:
          mean_c = qmu.mean[c]
          cov_c  = (L_c L_c^T)^{-1} + Cov_qmu[c]
        Returns a batch of K MultivariateNormals.
        """
        qmu: td.MultivariateNormal = self.qmu(**kwargs)
        mean = qmu.mean                         # (K, D)
        cov_qmu = qmu.covariance_matrix         # (K, D, D)

        # (L L^T)^{-1} via cholesky_solve on identity, then add Cov_qmu
        eye = torch.eye(self.latent_dim, device=self.L.device, dtype=self.L.dtype)
        cov_from_precision = torch.cholesky_solve(eye, self.L, upper=False)   # (K, D, D)
        cov = cov_from_precision + cov_qmu                                    # (K, D, D)

        # Build component Gaussians using scale_tril=chol(cov)
        L_cov = torch.linalg.cholesky(cov)                                     # (K, D, D)
        return td.MultivariateNormal(loc=mean, scale_tril=L_cov)

    # ----- log p(z | c) with expectation over q(mu) collapsed in closed form -----
    def log_prob_z_c(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute E_{q(mu)}[ log N(z | mu, (L L^T)^{-1}) ] per component, as in your TF code:
          = 0.5*log|P/(2π)| - 0.5*(z - E[mu])^T P (z - E[mu]) - 0.5*tr(P Cov_qmu)
        where P = L L^T is the precision matrix and Cov_qmu is q(mu)'s covariance.
        Returns (B, K).
        """
        qmu: td.MultivariateNormal = self.qmu(**kwargs)
        m = qmu.mean                              # (K, D)
        S = qmu.covariance_matrix                 # (K, D, D)

        # Precision per component: P = L L^T
        P = self.L @ self.L.transpose(-1, -2)     # (K, D, D)

        # Broadcast z to (B, K, D); m is (K, D)
        if z.dim() == 2:  # (B, D)
            z = z.unsqueeze(1)                    # (B, 1, D)
        # z: (B, 1, D) -> (B, K, D)
        z = z.expand(-1, self.num_clusters, -1)
        d = z - m.unsqueeze(0)                    # (B, K, D)

        # 0.5 * log|P / (2π)| = 0.5*(log|P| - D*log(2π))
        D = self.latent_dim
        logdetP = 2.0 * torch.sum(torch.log(torch.diagonal(self.L, dim1=-2, dim2=-1)), dim=-1)  # (K,)
        const = 0.5 * (logdetP - D * math.log(2.0 * math.pi))                                    # (K,)
        const = const.unsqueeze(0)                                                               # (1, K)

        # Mahalanobis term: -0.5 * d^T P d  -> (B,K)
        # Using einsum: (B,K,i) (K,i,j) (B,K,j) -> (B,K)
        quad = -0.5 * torch.einsum("bki,kij,bkj->bk", d, P, d)

        # Trace term: -0.5 * tr(P S) per component, then broadcast to (B,K)
        tr_term = -0.5 * torch.einsum("kij,kji->k", P, S)  # (K,)
        tr_term = tr_term.unsqueeze(0)                     # (1, K)

        expected_ll = const + quad + tr_term               # (B, K)
        return expected_ll