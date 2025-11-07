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

temp_decay = 0.00005
lr = 5e-4
epochs = 1500
num_train_data = len(dl)

abs_r_list = []
predict_label_nmi_list = []
predict_label_ari_list = []
gmm_fit_label_nmi_list = []
gmm_fit_label_ari_list = []

for trial in range(10):
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
            #loss, metrics = model(xb, max(ep/200.0, 1.0))
            loss, metrics = model(xb, 1.0)
            loss.backward()
            optimizer.step()
            # gather metrics
            #print("temperature", temp, step)
            for k,v in metrics.items():
                m_gather[k] = m_gather.get(k, 0.) + metrics[k]
        '''
        print("temperature", temp, step)
        for k,v in m_gather.items():
            m_gather[k] /= num_train_data
        print(f'train (average) : {m_gather}')
        '''
    model.eval()
    with torch.no_grad():
        latent_params = model.encoder(X.to(device))
        _,_,probs,_ = model.vector_quantization._posterior_quantize_probs(latent_params, model.vector_quantization._compute_var_q())
        predicted_clusters =  torch.argmax(probs, dim=1)
        np.save(save_path + "latent_params.npy", latent_params.detach().cpu().numpy())
        np.save(save_path + "probs.npy", probs.detach().cpu().numpy())
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