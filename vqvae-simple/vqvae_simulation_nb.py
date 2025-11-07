import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
#import utils
from vqvae import VQVAE

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
import math

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

datatype = "data2"
data_path = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/vampprior-mixture-model-test/origin/VampPrior-Mixture-Model/data/simulation/1d-nb/"+datatype+"/"
X = np.load(data_path + "X.npy")
y = np.load(data_path + "Y.npy")
Z_true = np.load(data_path + "Z.npy")
true_centers = np.load(data_path + "centers.npy")
n_clusters = len(true_centers)
X = torch.from_numpy(X).float()
dl = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)

epochs = 1000
lr = 1e-4

abs_r_list = []
predict_label_nmi_list = []
predict_label_ari_list = []
gmm_fit_label_nmi_list = []
gmm_fit_label_ari_list = []

for trial in range(10):
    save_path = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/vqvae-test/vqvae-simple/results/vqvae/simulation/1d-nb/"+datatype+"/trial_"+str(trial)+"/"
    os.makedirs(save_path, exist_ok=True)
    model = VQVAE(
        in_dim=100, x_dim=100, h_dim=64,
        n_layers_enc=2, n_layers_dec=2,
        n_embeddings=4, embedding_dim=1, beta=0.25,
        #embedding_init=np.array(cluster_centers).reshape(4,1),
        trainable_codes=True,
        likelihood="nb",
        #out_activation='sigmoid'
        use_ema = True,
        enable_reset = True,
        reset_interval = 50,        # check every N training steps
        usage_threshold = 1.0,     # codes below this EMA count are "dead"
        reinit_strategy = "samples", # "samples" or "gaussian"
        buffer_size = 512,          # recent z buffer
        #scalar_theta=True,
    ).to(device)
    results = {
        "n_updates": 0,
        "recon_loss": [],     # per step
        "loss_vals": [],      # per step (recon + embedding/commit)
        "perplexities": [],   # per step
    }
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n_update = 0
    for ep in range(1, epochs + 1):
        epoch_tot_loss = 0.0
        epoch_tot_recon = 0.0
        epoch_tot_perp  = 0.0
        epoch_count     = 0
        for (xb,) in dl:
            n_update += 1
            xb = xb.to(device).float()
            opt.zero_grad()
            # Your model should return: embedding/commitment loss, reconstruction x_hat, perplexity, indices
            #embedding_loss, x_hat, perplexity, indices = model(xb)
            #recon_loss = ((x_hat - xb) ** 2).mean()  # replace with your NB NLL if needed
            #recon_loss = ((x_hat - xb) ** 2).sum()/xb.shape[0]
            embedding_loss, recon_loss, perplexity, _ = model(xb)
            loss = recon_loss + embedding_loss
            loss.backward()
            opt.step()
            # ---- per-step logging ----
            results["recon_loss"].append(recon_loss.detach().item())
            results["perplexities"].append(perplexity.detach().item())
            results["loss_vals"].append(loss.detach().item())
            results["n_updates"] = n_update
            # ---- epoch aggregates (weighted by batch size) ----
            bsz = xb.size(0)
            epoch_tot_loss  += loss.detach().item() * bsz
            epoch_tot_recon += recon_loss.detach().item() * bsz
            epoch_tot_perp  += perplexity.detach().item() * bsz
            epoch_count     += bsz
        # ---- epoch summary print ----
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            mean_loss  = epoch_tot_loss / epoch_count
            mean_recon = epoch_tot_recon / epoch_count
            mean_perp  = epoch_tot_perp / epoch_count
            print(f"[{ep:03d}] loss={mean_loss:.3f}  recon={mean_recon:.3f}  perp={mean_perp:.2f}")
    model.eval()
    with torch.no_grad():
        vq_loss, x_hat, perplexity, predicted_clusters = model(X.float().to(device))
        latent_params = model.encoder(X.float().to(device))
    np.save(save_path + "latent_params.npy", latent_params.detach().cpu().numpy())
    #np.save(save_path + "probs.npy", probs.detach().cpu().numpy())
    np.save(save_path + "predicted_clusters.npy", predicted_clusters.detach().cpu().numpy())
    ari = adjusted_rand_score(y, predicted_clusters.cpu().numpy())
    nmi = normalized_mutual_info_score(y, predicted_clusters.cpu().numpy())
    enc_mu = latent_params.detach().cpu().numpy()
    r, p = pearsonr(enc_mu.reshape(-1), Z_true.reshape(-1))   # r is Pearson ρ estimate, p is two-sided p-value
    abs_r = abs(r)
    abs_r_list.append(abs_r)
    predict_label_nmi_list.append(nmi)
    predict_label_ari_list.append(ari)
    Z_latent = enc_mu.reshape(-1,1)
    model_gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=0).fit(Z_latent)
    resp_gmm = model_gmm.predict_proba(Z_latent)
    labels_gmm = model_gmm.predict(Z_latent)
    nmi = normalized_mutual_info_score(y, labels_gmm, average_method="arithmetic")
    ari = adjusted_rand_score(y, labels_gmm)
    gmm_fit_label_nmi_list.append(nmi)
    gmm_fit_label_ari_list.append(ari)