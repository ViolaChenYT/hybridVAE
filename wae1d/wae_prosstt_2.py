from wae import *
from torch.distributions import Categorical, Normal, MixtureSameFamily
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # v2 API
import torch.nn as nn
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os

idx_to_nodes = {
    1:"A",
    2:"B",
    3:"C",
    4:"D",
    5:"E",
    6:"F",
    7:"G",
    8:"H",
    9:"I",
    10:"J"
}


nodes_to_idx = {
    "A":1,
    "B":2,
    "C":3,
    "D":4,
    "E":5,
    "F":6,
    "G":7,
    "H":8,
    "I":9,
    "J":10
}

device = "cuda" if torch.cuda.is_available() else "cpu"

pearson_r_list = []
nmi_list = []
ari_list = []

for data_idx in range(100):
    data_path = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/simulation/prosstt/outputs/paths_2/"+str(data_idx)+"/"
    X = np.load(data_path + "count_mat.npy")
    y = np.load(data_path + "labels.npy")
    adata = sc.read("/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/simulation/prosstt/benchmarks/phate/paths_2/run_1/data_"+str(data_idx)+".h5ad")
    n_clusters = len(np.unique(y))
    pseudotime = np.load(data_path + "pseudotime.npy")
    true_centers = 15+30*np.array([i for i in range(n_clusters)])
    means = torch.from_numpy(true_centers).float()
    #means = torch.tensor([-20.0, -10.0, 0.0, 10.0])
    stds  = torch.tensor([1 for _ in range(n_clusters)])
    model = GMMWAE1D(
        means = means,
        stds = stds,
        n_clusters = n_clusters,
        #prior = gmm,
        device = device,
        in_dim=X.shape[1], x_dim=X.shape[1], h_dim=128,
        n_layers_enc=2, n_layers_dec=2,
        embedding_dim=1, 
        likelihood="nb",
        auto_init=0,
        update_prior_mix=0,
        tau = 0.2,
    ).to(device)
    phate_D = squareform(pdist(adata.obsm["X_phate_2d"]))
    k = 51
    # partition so that the k-th smallest element in each row is at index k-1
    kth_values = np.partition(phate_D, k-1, axis=1)[:, k-1]
    dataset = DistanceDataset(X, phate_D)
    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(dataset, batch_size=128, shuffle=True)
    D_full = dataset.D.to(device)
    epochs = 1500
    num_train_data = len(dl)
    frames = []
    train_loss_list = []
    train_recon_loss_list = []
    train_w2_list = []
    for ep in range(1, epochs + 1):
        print(f"epoch: {ep*1}/{epochs}")
        model.train()
        m_gather = {}
        idx = 0
        for xb, idx_b in dl:
            idx+=1
            xb = xb.to(device).float()
            D_batch = D_full[idx_b][:, idx_b].to(device)
            #xb_log1p = torch.log1p(xb)
            # update 
            optimizer.zero_grad()
            if ep<700:
                loss, metrics = model(xb,D_batch,lambda1 = 20,lambda2=1,threshold=kth_values.max(),weight = 0)
            else:
                loss, metrics = model(xb,D_batch,lambda1 = 0,lambda2=0.1,threshold=kth_values.max(),weight = 1)
            #loss, metrics = model(xb,D_batch,lambda1 = 20,lambda2=1,threshold=kth_values.max(),weight = 0)
            loss.backward()
            optimizer.step()
            # gather metrics
            #print("temperature", temp, step)
            for k,v in metrics.items():
                m_gather[k] = m_gather.get(k, 0.) + metrics[k]
        for k,v in m_gather.items():
            m_gather[k] /= num_train_data
        train_loss_list.append(m_gather["loss"].item())
        train_recon_loss_list.append(m_gather["recon_loss"].item())
        train_w2_list.append(m_gather["wasserstein_distance"].item())
        print(f'train (average) : {m_gather}')
    model.eval()
    with torch.no_grad():
        X_log1p = torch.log1p(torch.from_numpy(X).float())
        latent_params = model.encoder(X_log1p.to(device))
        enc_mu = latent_params.detach().cpu().numpy()
    save_path = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/codes/simulation/prosstt/benchmarks/wae/paths_2/run_2/"
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + "latent_z_"+str(data_idx)+".npy", enc_mu)
    y_int = np.array([nodes_to_idx[s] for s in y.ravel()], dtype=int).reshape(y.shape)
    r, p = pearsonr(enc_mu.reshape(-1), pseudotime.reshape(-1))   # r is Pearson ρ estimate, p is two-sided p-value
    abs_r = abs(r)
    print(abs_r)
    pearson_r_list.append(abs_r)
    Z_latent = enc_mu.reshape(-1,1)
    model_gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=0).fit(Z_latent)
    resp_gmm = model_gmm.predict_proba(Z_latent)
    labels_gmm = model_gmm.predict(Z_latent)
    nmi = normalized_mutual_info_score(y_int, labels_gmm, average_method="arithmetic")
    ari = adjusted_rand_score(y_int, labels_gmm)
    nmi_list.append(nmi)
    ari_list.append(ari)