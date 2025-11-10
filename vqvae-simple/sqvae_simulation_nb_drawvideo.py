import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
#import utils
from vqvae import VQVAE, SQVAE

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
import math
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
import os

import imageio.v2 as imageio
from torch.distributions import Categorical, Normal, MixtureSameFamily

device = "cuda" if torch.cuda.is_available() else "cpu"

datatype = "data1"
data_path = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/vampprior-mixture-model-test/origin/VampPrior-Mixture-Model/data/simulation/1d-nb/"+datatype+"/"
X = np.load(data_path + "X.npy")
y = np.load(data_path + "Y.npy")
Z_true = np.load(data_path + "Z.npy")
true_centers = np.load(data_path + "centers.npy")
n_clusters = len(true_centers)

X = torch.from_numpy(X).float()
dl = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)

temp_decay = 0.00005
lr = 5e-4
epochs = 1500
num_train_data = len(dl)

abs_r_list = []
predict_label_nmi_list = []
predict_label_ari_list = []
gmm_fit_label_nmi_list = []
gmm_fit_label_ari_list = []

for trial in range(5,15):
    save_path = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/vqvae-test/vqvae-simple/results/sqvae/simulation/1d-nb/"+datatype+"/trial_"+str(trial)+"/"
    os.makedirs(save_path, exist_ok=True)
    model = SQVAE(
        in_dim=100, x_dim=100, h_dim=64,
        n_layers_enc=2, n_layers_dec=2,
        n_embeddings=4, embedding_dim=1, 
        var_x_init=0.5,
        var_q_init = 1.0,
        likelihood = "nb",
        #beta=0.25,
        #embedding_init=np.array(cluster_centers).reshape(4,1),
        #trainable_codes=True,
        #out_activation='sigmoid'
        #use_ema = True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    frames = []
    for ep in range(1, epochs + 1):
        print(f"epoch: {ep*1}/{epochs}")
        model.train()
        m_gather = {}
        idx = 0
        for (xb,) in dl:
            idx+=1
            xb = xb.to(device).float()
            # temperature annealing
            step = idx + ep * num_train_data
            temp = np.max([np.exp(- temp_decay * step), 1e-5])
            model.set_temperature(temp)
            # update 
            optimizer.zero_grad()
            loss, metrics = model(xb, max(ep/200.0, 1.0))
            #loss, metrics = model(xb, 1.0)
            loss.backward()
            optimizer.step()
            # gather metrics
            #print("temperature", temp, step)
            for k,v in metrics.items():
                m_gather[k] = m_gather.get(k, 0.) + metrics[k]
        if ep%10 ==1:
            model.eval()
            with torch.no_grad():
                X_log1p = torch.log1p(X)
                latent_params = model.encoder(X_log1p.to(device))
                enc_mu = latent_params.detach().cpu().numpy()
            mix_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
            # component means and stds (K,)
            #means = torch.from_numpy(true_centers.reshape(-1)).float()
            #means = torch.tensor([-20.0, -10.0, 0.0, 10.0])
            means = model.vector_quantization.embedding.data.reshape(-1).detach().cpu()
            var_q = model.vector_quantization._compute_var_q().detach().cpu()  # scalar tensor
            std  = torch.sqrt(var_q)                                          # tensor scalar
            stds = std.repeat(n_clusters)     
            # define distributions
            mix = Categorical(mix_probs)
            comp = Normal(means, stds)               # batch of K Normals
            gmm = MixtureSameFamily(mix, comp)
            gmm_device = gmm.component_distribution.loc.device
            # or, more generally:
            # gmm_device = gmm.mixture_distribution.logits.device
            # 1D case
            xs = torch.linspace(
                means.min()-2*std.item(),
                means.max()+2*std.item(),
                500,
                device=gmm_device,
            )
            logprob = gmm.log_prob(xs)          # same device as gmm
            pdf = torch.exp(logprob)
            bins = np.histogram_bin_edges(enc_mu, bins=50)   # or "auto"
            left  = bins[:-1]
            width = np.diff(bins)
            # overall counts
            total_counts, _ = np.histogram(enc_mu, bins=bins)
            # normalizing constant so sum(density * width) = 1
            area = np.sum(total_counts * width)   # == len(enc_mu) if equal-width bins
            # overall density (outline)
            total_density = total_counts / area
            fig, ax = plt.subplots(figsize=(6, 4))
            # outline of overall density (no fill)
            ax.bar(left,
                total_density,
                width=width,
                edgecolor='k',
                fill=False,
                linewidth=1.2,
                label='All')
            # mixture pdf curve (already in density scale)
            ax.plot(xs, pdf, linewidth=2)
            # per-label stacked densities
            bottom = np.zeros_like(total_density, dtype=float)
            for lab in np.unique(y):
                sel = (y == lab)
                counts, _ = np.histogram(enc_mu[sel], bins=bins)
                density = counts / area                         # same normalizer
                ax.bar(left,
                    density,
                    width=width,
                    bottom=bottom,
                    label=str(lab),
                    alpha=0.7)
                bottom += density
            ax.set_xlabel('Encoded Latent Value')
            ax.set_ylabel('Density')
            ax.set_title('Encoded Latent Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            #plt.show()
            #plt.close()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            #frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            #frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame = buf.reshape(h, w, 4)[..., :3]   # drop alpha → RGB
            frames.append(frame)
            plt.close(fig)
    if frames:
        imageio.mimsave(
            "latent_hist_evolution_sqvae_10_1500_"+str(trial)+".mp4",
            frames,
            fps=5,
            macro_block_size=None,
        )
        print("Saved video to latent_hist_evolution.mp4")
    else:
        print("No frames collected; check your epoch condition.")