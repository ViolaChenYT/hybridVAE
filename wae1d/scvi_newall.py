import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
import anndata
import networkx as nx
import sys
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.distributions import Categorical, Normal, MixtureSameFamily, kl_divergence
import math
import json
import phate
from scipy.spatial.distance import pdist, squareform
from itertools import product

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)

def expand_choices(items, sep="/", strip=True, dedup=False):
    """
    items: list[str] possibly containing 'a/b' meaning choose one
    returns: list[list[str]] of all expanded lists
    """
    # turn each item into a list of options
    opts = []
    for s in items:
        parts = s.split(sep)
        if strip:
            parts = [p.strip() for p in parts]
        # if no sep, keep as single option; if sep, use all parts
        opts.append(parts if len(parts) > 1 else [parts[0]])
    # Cartesian product over positions
    combos = [list(choice) for choice in product(*opts)]
    if dedup:  # remove duplicate lists if options had duplicates
        seen = set()
        uniq = []
        for c in combos:
            t = tuple(c)
            if t not in seen:
                seen.add(t); uniq.append(c)
        return uniq
    return combos

def is_prefix_match(prefix: str, s: str, wildcard: str = "x") -> bool:
    """Return True iff prefix (with 'x' as single-char wildcard) matches the start of s."""
    if len(prefix) > len(s):
        return False
    for a, b in zip(prefix, s):
        if a != wildcard and b!=wildcard and a != b:
            return False
    return True

def is_prefix_chain_wild(L, wildcard: str = "x") -> bool:
    """Check L[i] is a wildcard-prefix of L[i+1] for all i."""
    return all(is_prefix_match(L[i], L[i+1], wildcard) for i in range(len(L) - 1))

def label_order_index(label_list, sep="/", wildcard="x"):
    # Find the first expanded path whose length-lex order forms a valid wildcard-prefix chain
    path = next(
        (sorted(p, key=lambda s: (len(s), s))
         for p in expand_choices(label_list, sep=sep)
         if is_prefix_chain_wild(sorted(p, key=lambda s: (len(s), s)), wildcard)),
        None
    )
    if path is None:
        return {}, 0
    label_idx = {lab: len(lab) for lab in path}
    #print(label_idx)
    #print(path)
    missing_node = int(len(label_idx) < len(path[-1])-len(path[0])+1)  # keep original rule
    # Map each original token like "a/b" to the first option that appears in label_idx
    origin = {}
    for lab in label_list:
        val = next((label_idx[o] for o in lab.split(sep) if o in label_idx), None)
        if val is not None:
            origin[lab] = val
    return origin, missing_node

def wasserstein_distance_1d_mixture_sample(
    encoded_samples: torch.Tensor,
    centroids: torch.Tensor,
    log_stds: torch.Tensor,
    mix_logits: torch.Tensor | None = None,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fast stochastic approximation of 1D W2^2 between:
      - empirical distribution of encoded_samples (uniform over batch),
      - Gaussian mixture prior with:
          means       = centroids [K]
          stds        = softplus(log_stds) [K]
          mix weights = softmax(mix_logits) if provided, else uniform.

    Strategy:
      - Draw B samples from the mixture prior using a Gumbel-Softmax relaxation
        (B = batch size).
      - Sort both sets and compute mean squared difference between order stats.

    Complexity: O(B K) + O(B log B), with B = batch_size, K = #components.
    """
    # flatten encoded_samples -> [B]
    z = encoded_samples.view(-1)
    B = z.size(0)
    device = z.device
    dtype = z.dtype

    K = centroids.shape[0]
    assert log_stds.shape[0] == K

    # stds > 0
    stds = F.softplus(log_stds) + eps  # [K]

    # mixture logits / probabilities
    if mix_logits is None:
        logits = torch.zeros(K, device=device, dtype=dtype)
    else:
        logits = mix_logits.to(device=device, dtype=dtype)

    # Gumbel-Softmax reparameterization for mixture assignment
    # sample Gumbel noise
    tiny = torch.finfo(dtype).tiny
    U = torch.rand(B, K, device=device, dtype=dtype)
    U = U.clamp(min=tiny, max=1.0 - tiny)
    g = -torch.log(-torch.log(U))  # [B, K]

    # relaxed one-hot weights for components, per sample
    y = F.softmax((logits.unsqueeze(0) + g) / tau, dim=-1)  # [B, K]

    # base Normal noise for each component/sample
    eps_base = torch.randn(B, K, device=device, dtype=dtype)  # [B, K]
    comp_samples = centroids.view(1, K) + stds.view(1, K) * eps_base  # [B, K]

    # mixture samples: convex combination of component samples
    prior_samples = (y * comp_samples).sum(dim=-1)  # [B]

    # 1D W2^2 approximation via sorted order statistics
    z_sorted, _ = torch.sort(z)
    prior_sorted, _ = torch.sort(prior_samples)
    w2 = torch.mean((z_sorted - prior_sorted) ** 2)
    return w2


def wasserstein_distance_1d_learnable(
    encoded_samples: torch.Tensor,
    centroids: torch.Tensor,
    log_stds: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    1D approximate W2 distance between:
      - empirical distribution of encoded_samples
      - Gaussian mixture with fixed centroids and learnable stds
    centroids: [K]
    log_stds: [K] (nn.Parameter); stds = softplus(log_stds)
    """
    # Flatten encoded_samples to [B, 1]
    z = encoded_samples.view(-1, 1)
    B = z.size(0)
    device = z.device
    dtype = z.dtype
    K = centroids.shape[0]
    assert log_stds.shape[0] == K

    # positive stds
    stds = F.softplus(log_stds) + eps  # [K]

    # --- Build deterministic "prior samples" ---
    # Use m quantiles from N(0,1) per component, then shift/scale
    m = math.ceil(B / K)
    u = (torch.arange(m, device=device, dtype=dtype) + 0.5) / m  # [m]
    base = Normal(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    ).icdf(u)  # [m]
    base = base.view(m, 1)  # [m, 1]

    prior_samples_list = []
    for k in range(K):
        mu_k = centroids[k]         # scalar
        std_k = stds[k]             # scalar
        prior_k = mu_k + std_k * base  # [m, 1]
        prior_samples_list.append(prior_k)

    prior_samples = torch.cat(prior_samples_list, dim=0)  # [K*m, 1]

    # Match number of samples to B
    if prior_samples.size(0) > B:
        prior_samples = prior_samples[:B]
    elif prior_samples.size(0) < B:
        repeat_times = math.ceil(B / prior_samples.size(0))
        prior_samples = prior_samples.repeat(repeat_times, 1)[:B]

    # --- Sort both and compute 1D W2 (MSE between quantiles) ---
    z_sorted, _ = torch.sort(z, dim=0)
    prior_sorted, _ = torch.sort(prior_samples, dim=0)
    wd = F.mse_loss(z_sorted, prior_sorted, reduction="sum") / B
    return wd

def pairwise_distance_loss(
    z: torch.Tensor,
    batch_indices: torch.Tensor,
    dist_mat: torch.Tensor,
    dist_scale: torch.Tensor = None,  # ==== NEW ====
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    z: [B, latent_dim]
    batch_indices: [B] global indices of the cells in this batch
    dist_mat: [N, N] precomputed pairwise distances between cells (no grad)
    dist_scale: scalar multiplier applied to dist_mat (learnable)
    """
    z_flat = z.view(z.size(0), -1)  # [B, latent_dim]

    # pairwise Euclidean distances in latent space
    d_latent = torch.cdist(z_flat, z_flat, p=2)  # [B, B]

    # corresponding target distances for this batch
    idx = batch_indices.long()
    d_target = dist_mat[idx][:, idx]  # [B, B]

    # ==== NEW: apply learnable scaling ====
    if dist_scale is not None:
        d_target = d_target * dist_scale
    # ======================================

    if normalize:
        d_latent = d_latent / (d_latent.mean() + eps)
        d_target = d_target / (d_target.mean() + eps)

    return F.mse_loss(d_latent, d_target)

def mixture_uniform_reg(mix_logits: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation of mixture weights π from uniform.
    π = softmax(mix_logits); u_k = 1/K.

    Returns sum_k (π_k - 1/K)^2.
    """
    pi = torch.softmax(mix_logits, dim=0)
    K = pi.size(0)
    uniform = pi.new_full((K,), 1.0 / K)
    return torch.sum((pi - uniform) ** 2)

def gmm_cluster_1d(X, means, stds, mix_probs, eps=1e-8):
    """
    Cluster 1D data X using a Gaussian mixture model with *fixed* parameters,
    and compute log p(x_i) for each data point.
    Parameters
    ----------
    X : array-like, shape (N,)
        Data points.
    means : array-like, shape (K,)
        Means of each Gaussian component.
    stds : array-like, shape (K,)
        Standard deviations of each Gaussian component.
    mix_probs : array-like, shape (K,)
        Mixture weights (should sum to 1).
    Returns
    -------
    labels : ndarray, shape (N,)
        Hard cluster assignments (0..K-1).
    responsibilities : ndarray, shape (N, K)
        Posterior probabilities p(z=k | x_i).
    log_px : ndarray, shape (N,)
        Log-likelihood log p(x_i) under the mixture model.
    """
    X = np.asarray(X, dtype=float)          # (N,)
    means = np.asarray(means, dtype=float)  # (K,)
    stds = np.asarray(stds, dtype=float)    # (K,)
    mix_probs = np.asarray(mix_probs, dtype=float)  # (K,)
    # safety
    stds = np.maximum(stds, eps)
    mix_probs = mix_probs / mix_probs.sum()
    N = X.shape[0]
    K = means.shape[0]
    # reshape for broadcasting:
    # X: (N, 1), means/stds/mix_probs: (1, K)
    X_expanded = X[:, None]        # (N, 1)
    means_expanded = means[None]   # (1, K)
    stds_expanded = stds[None]     # (1, K)
    mix_expanded = mix_probs[None] # (1, K)
    var = stds_expanded ** 2
    # log p(x_i, z=k) = log pi_k + log N(x_i | mu_k, sigma_k^2)
    log_norm_const = -0.5 * np.log(2.0 * np.pi * var)               # (1, K)
    log_exp_term = -0.5 * (X_expanded - means_expanded) ** 2 / var  # (N, K)
    log_comp = log_norm_const + log_exp_term                        # (N, K)
    log_joint = np.log(mix_expanded + eps) + log_comp               # (N, K)
    # log p(x_i) via log-sum-exp over k
    log_joint_max = np.max(log_joint, axis=1, keepdims=True)        # (N, 1)
    joint_shifted = np.exp(log_joint - log_joint_max)               # (N, K)
    sum_joint = np.sum(joint_shifted, axis=1, keepdims=True)        # (N, 1)
    log_px = (log_joint_max + np.log(sum_joint + eps)).ravel()      # (N,)
    # responsibilities p(z=k | x_i)
    responsibilities = joint_shifted / (sum_joint + eps)            # (N, K)
    # Hard assignments
    labels = np.argmax(responsibilities, axis=1)                    # (N,)
    return labels, responsibilities, log_px


if __name__ == '__main__':
    ### load data
    adata = sc.read("/n/fs/ragr-data/users/viola/structuredVAE/data/scvi_path_ABprp_ABprppaaaapp/trained.h5ad")

    if hasattr(adata.X, "toarray"):
        X_np = adata.X.toarray()
    else:
        X_np = np.asarray(adata.X)

    count_mat_tensor = torch.from_numpy(X_np).float()
    N_tot = count_mat_tensor.size(0)  # ==== NEW: use this early ====
    unique_batches, batch_int = np.unique(np.asarray(adata.obs["batch"].tolist()), return_inverse=True)
    batch_int_tensor = torch.from_numpy(batch_int).to(device).to(torch.int64).unsqueeze(-1)
    label_idx_dict, if_missing = label_order_index(adata.obs["lineage"].unique().tolist())
    print(label_idx_dict)
    lineage_label = adata.obs['lineage'].apply(
                lambda x: label_idx_dict[x]
            ).tolist()
    n_clusters = len(np.unique(np.array(lineage_label)))
    #n_clusters = len(np.unique(np.array(adata.obs["lineage"].tolist())))
    #labels = np.array(adata.obs["lineage"].tolist())


    ### compute phate to get pairwise distance matrix
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)
    #sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata_copy, max_value=10)
    sc.tl.pca(adata_copy, svd_solver="arpack")
    phate_operator = phate.PHATE()
    _ = phate_operator.fit_transform(adata_copy.obsm['X_pca'])
    pairwise_dist_np = squareform(pdist(phate_operator.diff_potential, "euclidean"))
    pairwise_dist_tensor = torch.from_numpy(pairwise_dist_np).to(device).float()

    ### initialize scvi
    if hasattr(adata.X, "tocsr"):
        adata.layers["counts"] = adata.X.copy().tocsr()
    else:
        adata.layers["counts"] = adata.X.copy()

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
        n_layers=2,
        n_hidden=64,
        n_latent=1,
    )
    vae = scvi.model.SCVI(adata, **arches_params)
    model = vae.module.to(device)

    ### initialize data loader
    indices = torch.arange(N_tot)
    dataset = TensorDataset(count_mat_tensor, indices)
    dl = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=False,
    )

    ### learnable parameters in prior (tunable!!)
    train_prior_means = True
    train_prior_stds = True
    train_prior_mix = False

    centroids = torch.nn.Parameter(torch.zeros(n_clusters, device=device))
    log_stds = torch.nn.Parameter(torch.zeros(n_clusters, device=device))
    mix_logits = torch.nn.Parameter(torch.zeros(n_clusters, device=device))

    centroids.requires_grad = train_prior_means
    log_stds.requires_grad = train_prior_stds
    mix_logits.requires_grad = train_prior_mix

    ### learnable scale for the distance loss
    log_dist_scale = torch.nn.Parameter(torch.tensor(0.0, device=device))
    ### initialize optimizer
    opt_ae = torch.optim.Adam(
        list(model.parameters()) + [log_dist_scale],
        lr=1e-3,
        weight_decay=1e-5,
    )

    prior_params = []
    if train_prior_means:
        prior_params.append(centroids)
    if train_prior_stds:
        prior_params.append(log_stds)
    if train_prior_mix:
        prior_params.append(mix_logits)

    opt_prior = (
        torch.optim.Adam(prior_params, lr=1e-3, weight_decay=1e-5)
        if prior_params
        else None
    )

    def initialize_prior_from_encoder():
        """Initialize centroids / stds from aggregated posterior mean of current encoder."""
        model.eval()
        with torch.no_grad():
            X_tot = count_mat_tensor.to(device)
            N = X_tot.size(0)
            batch_all = torch.zeros((N, 1), device=device, dtype=torch.long)
            labels_all = torch.zeros((N, 1), device=device, dtype=torch.long)
            tensors_all = {"X": X_tot, "batch": batch_all, "labels": labels_all}
            inf_inputs_all = model._get_inference_input(tensors_all)
            inf_outputs_all = model._regular_inference(**inf_inputs_all)
            mu_q = inf_outputs_all["qz"].mean  # [N, 1]

            mu = mu_q.detach().flatten().cpu().numpy()
            mu_sort = np.sort(mu)

            means = []
            stds = []
            for k in range(n_clusters):
                start = int(k / n_clusters * len(mu_sort))
                end = int((k + 1) / n_clusters * len(mu_sort))
                data_batch = mu_sort[start:end]
                if data_batch.size == 0:
                    data_batch = mu_sort
                means.append(float(np.median(data_batch)))
                stds.append(float(np.std(data_batch)))

            means_t = torch.tensor(means, device=device, dtype=torch.float32)
            stds_t = torch.tensor(stds, device=device, dtype=torch.float32)
            stds_t = torch.clamp(stds_t, min=1e-3)

            # inverse softplus to initialize log_stds:
            # std = softplus(log_std)  =>  log_std = log(exp(std) - 1)
            log_stds_init = torch.log(torch.exp(stds_t) - 1.0)
            centroids.data.copy_(means_t)
            log_stds.data.copy_(log_stds_init)

        model.train()

    initialize_prior_from_encoder()

    ### training loop
    # hypere-parameters (tunable!!))
    epochs = 500
    kl_weight = 0.0
    wd_weight = 10.0
    dist_weight = 1.0
    tau_gumbel = 1.0
    mix_uniform_reg_weight = 1.0

    losses_history = []

    for ep in range(1, epochs + 1):
        model.train()
        epoch_ae = {"loss": 0.0, "recon_loss": 0.0, "wd": 0.0, "dist": 0.0}
        for xb, idx in dl:
            xb = xb.to(device).float()
            idx = idx.to(device)  # global indices for this batch

            bsz = xb.size(0)
            batch_tmp = batch_int_tensor[idx.long()]
            labels_tmp = torch.zeros((bsz, 1), device=device, dtype=torch.long)
            tensors = {"X": xb, "batch": batch_tmp, "labels": labels_tmp}

            opt_ae.zero_grad(set_to_none=True)
            inference_inputs = model._get_inference_input(tensors)
            inference_outputs = model._regular_inference(**inference_inputs)
            generative_inputs = model._get_generative_input(
                tensors=tensors, inference_outputs=inference_outputs
            )
            generative_outputs = model.generative(**generative_inputs)
            loss_info = model.loss(
                tensors=tensors,
                inference_outputs=inference_outputs,
                generative_outputs=generative_outputs,
                kl_weight=kl_weight,
            )
            recon_loss = loss_info.reconstruction_loss["reconstruction_loss"].mean()
            if kl_weight > 0.0:
                loss = loss_info.loss
            else:
                loss = recon_loss
            
            ### wasserstein distance to GMM prior
            if train_prior_mix:
                wd_ae = wasserstein_distance_1d_mixture_sample(
                    inference_outputs["z"],
                    centroids.detach(),
                    log_stds.detach(),
                    mix_logits.detach(),
                    tau=tau_gumbel,
                )
            else:
                wd_ae = wasserstein_distance_1d_learnable(
                    inference_outputs["z"],
                    centroids.detach(),
                    log_stds.detach(),
                )
            
            ### pairwise distance loss
            dist_scale = F.softplus(log_dist_scale) + 1e-6
            dist_loss = pairwise_distance_loss(
                inference_outputs["z"],
                idx,
                pairwise_dist_tensor,
                dist_scale=dist_scale,
                normalize=False,  # set True if you want scale invariance
            )

            ### total AE loss
            loss_ae = loss + wd_weight * wd_ae + dist_weight * dist_loss
            loss_ae.backward()
            opt_ae.step()

            epoch_ae["loss"] += loss_ae.item() * bsz
            epoch_ae["recon_loss"] += recon_loss.item() * bsz
            epoch_ae["wd"] += wd_ae.item() * bsz
            epoch_ae["dist"] += dist_loss.item() * bsz  # ==== NEW ====

        for k in epoch_ae:
            epoch_ae[k] /= float(N_tot)
        
        # ==== Prior update step ====
        if opt_prior is not None and wd_weight >0:
            model.eval()
            epoch_prior = {"loss": 0.0, "wd": 0.0}
            for xb, idx in dl:
                xb = xb.to(device).float()
                idx = idx.to(device)

                bsz = xb.size(0)
                batch_tmp = batch_int_tensor[idx.long()]
                labels_tmp = torch.zeros((bsz, 1), device=device, dtype=torch.long)
                tensors = {"X": xb, "batch": batch_tmp, "labels": labels_tmp}

                # compute z with encoder frozen
                with torch.no_grad():
                    inference_inputs = model._get_inference_input(tensors)
                    inference_outputs = model._regular_inference(**inference_inputs)
                    z_detached = inference_outputs["z"].detach()
                
                opt_prior.zero_grad(set_to_none=True)

                if train_prior_mix:
                    wd_prior = wasserstein_distance_1d_mixture_sample(
                        z_detached,
                        centroids,
                        log_stds,
                        mix_logits,
                        tau=tau_gumbel,
                    )
                else:
                    wd_prior = wasserstein_distance_1d_learnable(
                        z_detached,
                        centroids,
                        log_stds,
                    )
                # regularize mixture weights toward uniform (only matters if train_prior_mix=True)
                if mix_uniform_reg_weight > 0.0 and train_prior_mix:
                    reg_mix = mixture_uniform_reg(mix_logits)
                else:
                    reg_mix = z_detached.new_zeros(())
                ### total prior loss
                loss_prior = wd_weight * wd_prior + mix_uniform_reg_weight * reg_mix
                loss_prior.backward()
                opt_prior.step()

                epoch_prior["loss"] += loss_prior.item() * bsz
                epoch_prior["wd"] += wd_prior.item() * bsz

            for k in epoch_prior:
                epoch_prior[k] /= float(N_tot)

        # store & log
        if opt_prior is not None and wd_weight >0:
            losses_history.append(
                {
                    "ae_loss": epoch_ae["loss"],
                    "ae_recon_loss": epoch_ae["recon_loss"],
                    "ae_wd": epoch_ae["wd"],
                    "ae_dist": epoch_ae["dist"],      # ==== NEW ====
                    "prior_loss": epoch_prior["loss"],
                    "prior_wd": epoch_prior["wd"],
                }
            )
        else:
            losses_history.append(
                {
                    "ae_loss": epoch_ae["loss"],
                    "ae_recon_loss": epoch_ae["recon_loss"],
                    "ae_wd": epoch_ae["wd"],
                    "ae_dist": epoch_ae["dist"],      # ==== NEW ====
                }
            )

        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(
                f"[{ep:03d}] "
                f"AE: loss={epoch_ae['loss']:.3f}, recon={epoch_ae['recon_loss']:.3f}, "
                f"wd={epoch_ae['wd']:.3f}, dist={epoch_ae['dist']:.3f} | "
                #f"Prior: loss={epoch_prior['loss']:.3f}, wd={epoch_prior['wd']:.3f}"
            )
            if opt_prior is not None and wd_weight >0:
                print(
                    f"      Prior: loss={epoch_prior['loss']:.3f}, wd={epoch_prior['wd']:.3f}"
                )
    
    ### after training analysis
    model.eval()
    with torch.no_grad():
        X_tot = count_mat_tensor.to(device)
        #batch_tmp = torch.zeros((len(X_tot),1)).to(device)
        #batch_tmp.to(device)
        labels_tmp = torch.zeros((len(X_tot),1)).to(device)
        #labels_tmp.to(device)
        tensors = {"X":X_tot, "batch":batch_int_tensor.to(device), "labels":labels_tmp}
        inference_inputs = model._get_inference_input(tensors)
        inference_outputs = model._regular_inference(**inference_inputs)
        mu_q = inference_outputs["qz"].mean
        mu = mu_q.detach().flatten().to("cpu").numpy()
    r, p = spearmanr(mu.reshape(-1), np.array(lineage_label))
    abs_r = abs(r)
    print(abs_r)

    ### two different ways to compute NMI/ARI
    inbuilt_labels, inbuilt_resp, _ = gmm_cluster_1d(mu.reshape(-1),centroids.detach().to("cpu").numpy(),torch.exp(log_stds).detach().to("cpu").numpy(),torch.softmax(mix_logits,dim=0).detach().to("cpu").numpy())
    nmi = normalized_mutual_info_score(np.array(lineage_label), inbuilt_labels, average_method="arithmetic")
    ari = adjusted_rand_score(np.array(lineage_label), inbuilt_labels)
    print(nmi,ari)

    model_gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=0).fit(mu.reshape(-1,1))
    resp_gmm = model_gmm.predict_proba(mu.reshape(-1,1))
    labels_gmm = model_gmm.predict(mu.reshape(-1,1))
    nmi = normalized_mutual_info_score(np.array(lineage_label), labels_gmm, average_method="arithmetic")
    ari = adjusted_rand_score(np.array(lineage_label), labels_gmm)
    print(nmi,ari)

