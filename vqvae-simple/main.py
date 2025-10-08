import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
#import utils
from vqvae import VQVAE

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

@torch.no_grad()
def generate_clustered_gauss_1d(
    n_clusters: int = 4,
    points_per_cluster: int = 300,
    n_features: int = 100,
    latent_dim: int = 1,          # 1D latent space
    sigma_val: float = 0.5,       # observation noise std for Gaussian in feature space
    cluster_centers=None,         # if None, auto-generate
    cluster_spread: float = 0.2,  # smaller => tighter clusters in latent space
    seed: int = 42,
    device: str = "cpu",
):
    """
    Build a clean clusterable dataset with 1D latent space:
      z | c ~ N(center[c], cluster_spread^2) in R^latent_dim
      x | z ~ N(W z + b, sigma^2 I) in R^{n_features}
    Returns:
      X [N, D] (float), labels [N] (long), true Z [N, latent_dim], centers [K, latent_dim]
    """
    g = torch.Generator(device=device).manual_seed(seed)
    N = n_clusters * points_per_cluster

    # Set cluster centers in latent space (R^{latent_dim})
    if cluster_centers is None:
        # Evenly spaced along the first latent axis; zeros elsewhere if latent_dim>1
        centers_1d = torch.linspace(-2, 2, n_clusters, device=device)
        centers = torch.zeros(n_clusters, latent_dim, device=device)
        centers[:, 0] = centers_1d
    else:
        centers = torch.as_tensor(cluster_centers, dtype=torch.float32, device=device)
        if centers.ndim == 1:
            centers = centers.unsqueeze(1)  # [K,1] if given as list of scalars
        assert centers.shape == (n_clusters, latent_dim), \
            f"centers must be shape ({n_clusters},{latent_dim})"

    # Cluster assignments
    labels = torch.arange(n_clusters, device=device).repeat_interleave(points_per_cluster)  # [N]

    # Latent samples
    z = torch.randn(N, latent_dim, generator=g, device=device) * cluster_spread
    z += centers[labels]  # shift by cluster centers

    # Linear map to feature space
    W = torch.randn(n_features, latent_dim, generator=g, device=device) / math.sqrt(latent_dim)
    b = torch.randn(n_features, generator=g, device=device) * 0.2

    mean_x = z @ W.T + b  # [N, D]

    # Gaussian observation noise (isotropic)
    if sigma_val <= 0:
        X = mean_x
    else:
        X = mean_x + sigma_val * torch.randn(mean_x.shape, device=mean_x.device, generator=g)

    return X, labels.to("cpu"), z.to("cpu"), centers.to("cpu")


model = VQVAE(
    in_dim=None, x_dim=100, h_dim=64,
    n_layers_enc=2, n_layers_dec=2,
    n_embeddings=4, embedding_dim=1, beta=0.25,
    #embedding_init=init_w, normalize_init=True, trainable_codes=True,
    #out_activation='sigmoid'
).to(device)

lr = 3e-4
opt = optim.Adam(model.parameters(), lr=lr)
#criterion = nn.MSELoss()

dl = DataLoader(TensorDataset(X.float()), batch_size=128, shuffle=True)

epochs = 500

results = {
    'n_updates': 0,
    'recon_loss': [],
    'loss_vals': [],
    'perplexities': [],
}

n_update = 0
for ep in range(1, epochs+1):
    tot = 0.0; n = 0
    epoch_losses = {}
    
    for (xb,) in dl:
        n_update+=1
        xb = xb.to(device).float()
        opt.zero_grad()
        #losses = fit_step(model, xb, kl_w)
        embedding_loss, x_hat, perplexity = model(xb)
        recon_loss = torch.mean((x_hat - xb)**2)
        loss = recon_loss + embedding_loss
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        results["recon_loss"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        '''
        # Accumulate losses
        for key, value in losses.items():
            if key not in epoch_losses:
                epoch_losses[key] = 0.0
            epoch_losses[key] += value.item() * xb.size(0)
        '''
        
        tot += loss.item() * xb.size(0)
        n += xb.size(0)
    
    if ep == 1 or ep % 20 == 0 or ep == epochs:
        print(f"[{ep:03d}] loss={tot/n:.3f}")
    '''
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= n
    losses_history.append(epoch_losses)
    '''
    
    '''
    if ep == 1 or ep % 20 == 0 or ep == epochs:
        print(f"[{ep:03d}] loss={tot/n:.3f}  kl_w={kl_w:.2f}  recon={:.3f}  kl={epoch_losses.get('kl', 0):.3f}")
    '''

print("Training completed!")

for i in range(args.n_updates):
    (x, _) = next(iter(dl))
    x = x.to(device)
    optimizer.zero_grad()

    embedding_loss, x_hat, perplexity = model(x)
    recon_loss = torch.mean((x_hat - x)**2) / x_train_var
    loss = recon_loss + embedding_loss

    loss.backward()
    optimizer.step()

    results["recon_errors"].append(recon_loss.cpu().detach().numpy())
    results["perplexities"].append(perplexity.cpu().detach().numpy())
    results["loss_vals"].append(loss.cpu().detach().numpy())
    results["n_updates"] = i

    if i % args.log_interval == 0:
        """
        save model and print values
        """
        if args.save:
            hyperparameters = args.__dict__
            utils.save_model_and_results(
                model, results, hyperparameters, args.filename)

        print('Update #', i, 'Recon Error:',
                np.mean(results["recon_errors"][-args.log_interval:]),
                'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))