import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from minisom import MiniSom

# Paths
TIMESTAMP = "FEB_26_2026"
ROOT_DIR = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA"
INPUT_FILE = ROOT_DIR + "/OUTPUTS/Final/FINAL_GENIAL_DB.csv"
SILHOUETTE_PLOT_FILE = ROOT_DIR + f"/OUTPUTS/Plots/{TIMESTAMP}_SOM_silhouette.png"

BEHAVIORAL_VARS = [
    'SRS_social_cognition_tscore', 'SRS_social_communication_tscore',
    'SRS_restrictive_repetitive_tscore', 'attention_deficit_hyperactivity_tscore', 'nonverbal_iq'
]

# Load and preprocess
df = pd.read_csv(INPUT_FILE)
df = df[df['age_at_test'] < 19]
for col in BEHAVIORAL_VARS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=BEHAVIORAL_VARS).copy()

scaler = MinMaxScaler()
X = scaler.fit_transform(df[BEHAVIORAL_VARS].values)
n, p = X.shape
gx = gy = int(np.ceil(np.sqrt(int(np.ceil(5 * np.sqrt(n))))))
lin = lambda ij: ij[0] * gy + ij[1]
Ks = list(range(3, 10))  # K = 2 to 9


def fit_som(X, gx, gy, sigma=1.2, learning_rate=0.5, seed=42, iters=2000):
    som = MiniSom(x=gx, y=gy, input_len=X.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, num_iteration=iters, verbose=False)
    return som


def silhouette_per_K(X, som, bmus, Ks, lin):
    w = som.get_weights().reshape(-1, X.shape[1])
    out = []
    for K in Ks:
        lab = KMeans(n_clusters=K, random_state=0, n_init=20).fit(w).labels_
        sample_lab = np.array([lab[lin(ij)] for ij in bmus])
        out.append(silhouette_score(X, sample_lab))
    return out


# (1) sigma=1.2, lr from 0.3 to 0.5; (2) lr=0.5, sigma from 0.5 to 1.5
lrs_sigma_fixed = [0.3, 0.4, 0.5]
sigmas_sigma_varies = [0.5, 0.7, 1.0, 1.2, 1.5]
lr_sigma_varies = 0.5

results = {}  # (sigma, lr) -> list of silhouette per K
for lr in lrs_sigma_fixed:
    som = fit_som(X, gx, gy, sigma=1.2, learning_rate=lr)
    bmus = np.array([som.winner(x) for x in X])
    results[(1.2, lr)] = silhouette_per_K(X, som, bmus, Ks, lin)
for sigma in sigmas_sigma_varies:
    if (sigma, lr_sigma_varies) in results:
        continue
    som = fit_som(X, gx, gy, sigma=sigma, learning_rate=lr_sigma_varies)
    bmus = np.array([som.winner(x) for x in X])
    results[(sigma, lr_sigma_varies)] = silhouette_per_K(X, som, bmus, Ks, lin)

# Best (sigma, lr): max over K of silhouette
best_params = max(results.keys(), key=lambda k: max(results[k]))
best_sigma, best_lr = best_params
best_curve = results[best_params]

# Single figure: 3 panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# (1) sigma=1.2, lr varied
for lr in lrs_sigma_fixed:
    ax1.plot(Ks, results[(1.2, lr)], marker='o', label=f'lr={lr}')
ax1.set_xlabel('K')
ax1.set_ylabel('Silhouette score')
ax1.set_title('Sigma = 1.2 (vary learning rate)')
ax1.set_xticks(Ks)
ax1.legend()
ax1.grid(alpha=.3)

# (2) lr=0.5, sigma varied
for sigma in sigmas_sigma_varies:
    ax2.plot(Ks, results[(sigma, lr_sigma_varies)], marker='o', label=f'σ={sigma}')
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette score')
ax2.set_title('Learning rate = 0.5 (vary sigma)')
ax2.set_xticks(Ks)
ax2.legend()
ax2.grid(alpha=.3)

# (3) Best result
ax3.plot(Ks, best_curve, marker='o', color='black', linewidth=2, label=f'Best (σ={best_sigma}, lr={best_lr})')
ax3.set_xlabel('K')
ax3.set_ylabel('Silhouette score')
ax3.set_title('Best result')
ax3.set_xticks(Ks)
ax3.legend()
ax3.grid(alpha=.3)

plt.tight_layout()
plt.savefig(SILHOUETTE_PLOT_FILE, dpi=300)
plt.show()
print(f"Saved: {SILHOUETTE_PLOT_FILE}")
print(f"Best: sigma={best_sigma}, learning_rate={best_lr}, best K={Ks[np.argmax(best_curve)]}")

# Fit final SOM and cluster with best params
som = fit_som(X, gx, gy, sigma=best_sigma, learning_rate=best_lr)
bmus = np.array([som.winner(x) for x in X])
weights = som.get_weights().reshape(-1, p)
best_k = Ks[np.argmax(best_curve)]

km_final = KMeans(n_clusters=best_k, random_state=0, n_init=50).fit(weights)
sample_labels = np.array([km_final.labels_[lin(ij)] for ij in bmus])
centers = km_final.cluster_centers_
D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
cluster_probs = (1.0 / (1.0 + D)) / (1.0 / (1.0 + D)).sum(axis=1, keepdims=True)

# 2D scatter of participants (first two dimensions) colored by cluster,
# with K-means cluster centers overlaid (first two variables)
plt.figure(figsize=(6, 6))
for c in np.unique(sample_labels):
    plt.scatter(
        X[sample_labels == c, 0],
        X[sample_labels == c, 1],
        label=f"cluster={c}",
        alpha=0.7,
    )
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=120,
    linewidths=2,
    edgecolors="k",
    facecolors="none",
    label="cluster center",
)
plt.xlabel(BEHAVIORAL_VARS[0])
plt.ylabel(BEHAVIORAL_VARS[1])
plt.legend()
plt.tight_layout()
plt.show()

# Component planes: one SOM map per variable
W = som.get_weights()  # gx x gy x p
fig, axs = plt.subplots(1, p, figsize=(4 * p, 4))
if p == 1:
    axs = [axs]
for j, var in enumerate(BEHAVIORAL_VARS):
    plane = W[:, :, j].T
    im = axs[j].imshow(plane, origin="lower")
    axs[j].set_title(var)
    plt.colorbar(im, ax=axs[j], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Output
unique, counts = np.unique(sample_labels, return_counts=True)
cluster_means = scaler.inverse_transform(centers)
print("Participants per cluster:", dict(zip(unique, counts)))
df_out = df.copy()
df_out['cluster'] = sample_labels
for k in range(best_k):
    df_out[f'cluster_{k}_prob'] = cluster_probs[:, k]
print(f"Summary: grid {gx}x{gy}, K={best_k}, sil={silhouette_score(X, sample_labels):.4f}")
