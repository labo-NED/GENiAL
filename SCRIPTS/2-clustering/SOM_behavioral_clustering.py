import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
import os

# ------------------------------------------------------------
# PATHS & CONTSTANTS 
# ------------------------------------------------------------
TIMESTAMP = "MAR_09_2026"

ROOT_DIR = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA"
INPUT_FILE = ROOT_DIR + "/OUTPUTS/Final/FINAL_GENIAL_DB_without_cluster.csv"

OUTPUT_FILE = ROOT_DIR + f"/OUTPUTS/Clustered/clustered_SOM_Q1K_CHU_MHC_BC_DATA_{TIMESTAMP}.csv"
RADAR_PLOTS_FILE = ROOT_DIR + f"/OUTPUTS/Plots/SOM_Q1K_CHU_MHC_BC_DATA_{TIMESTAMP}_cluster_radars.png"
SELECTION_CURVES_FILE = ROOT_DIR + f"/OUTPUTS/Plots/SOM_Q1K_CHU_MHC_BC_DATA_{TIMESTAMP}_selection_curves.png"

BEHAVIORAL_VARS = [
    'SRS_social_cognition_tscore',
    'SRS_social_communication_tscore',
    'SRS_restrictive_repetitive_tscore',
    'attention_deficit_hyperactivity_tscore',
    # 'oppositional_defiant_tscore',
    'nonverbal_iq'
    # 'verbal_iq'
]
pretty_labels = {
    'SRS_social_cognition_tscore': 'Social Cognition',
    'SRS_social_communication_tscore': 'Social Communication',
    'SRS_restrictive_repetitive_tscore': 'Repetitive behavior',
    'attention_deficit_hyperactivity_tscore': 'ADHD',
    # 'oppositional_defiant_tscore': 'Oppositional',
    'nonverbal_iq': 'NVIQ'
    # 'verbal_iq': 'VIQ'
}
# -------------------------
# Load data
# -------------------------
df = pd.read_csv(INPUT_FILE)

# --- Code diagnosis ---
# Participants with all SRS columns present (complete SRS cases)
all_srs_notna = df[BEHAVIORAL_VARS].notna().all(axis=1)

n_total_complete = all_srs_notna.sum()

# Age filters
age_in = (df['age_at_test'] < 19)
age_out = ~age_in

# Complete SRS + in age range
n_in_age = (all_srs_notna & age_in).sum()
# Complete SRS + outside age range
n_out_age = (all_srs_notna & age_out).sum()

print(f"Number of participants with all SRS columns not empty: {n_total_complete}")
print(f"Number of participants with SRS and age below 18 inclusively: {n_in_age}")
print(f"Number of participants with SRS and age above 18: {n_out_age}")
if n_total_complete != (n_in_age + n_out_age):
    print(f"WHY? {n_total_complete} != {n_in_age} + {n_out_age} = {n_in_age + n_out_age}")

# --- END - Code diagnosis ---

# Keep participants with age between 0-18 inclusively
df = df[age_in]

# Ensure numeric and complete cases
for col in BEHAVIORAL_VARS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
complete_case_mask = df[BEHAVIORAL_VARS].notna().all(axis=1)
df = df.loc[complete_case_mask].copy()

# # Keep participants with diagnosis not empty
# df = df[df['diagnosis'].notna()]

# Scale 0-1 for SOM
scaler = MinMaxScaler()
X = scaler.fit_transform(df[BEHAVIORAL_VARS].values)

print(f"Data shape after preprocessing: {X.shape}")
print(f"Any NaN values remaining? {np.isnan(X).any()}")

# -------------------------
# Fit a single SOM grid based on n only
# -------------------------
n, p = X.shape
total_neurons = int(np.ceil(5 * np.sqrt(n)))
side = int(np.ceil(np.sqrt(total_neurons)))   # square grid
gx = gy = side

def fit_som(X, gx, gy, seed=42, iters=2000):
    som = MiniSom(x=gx, y=gy, input_len=X.shape[1],
                  sigma=1.5, learning_rate=0.5, random_seed=seed) # Chosen from parameter optimization
    som.random_weights_init(X)
    som.train_random(X, num_iteration=iters, verbose=False)
    return som

som = fit_som(X, gx, gy, seed=42, iters=2000)

# Diagnostics
qe = som.quantization_error(X)
te = som.topographic_error(X)
print(f"SOM {gx}x{gy}  QE={qe:.4f}  TE={te:.4f}")

# U-matrix
plt.figure(figsize=(6,6))
umat = som.distance_map()                   # gx x gy
plt.title("U-matrix")
plt.imshow(umat.T, origin="lower")
plt.colorbar()
plt.tight_layout()
plt.show()

# Hit map
hits = np.zeros((gx, gy), dtype=int)
bmus = []
for x in X:
    ij = som.winner(x)
    bmus.append(ij)
    hits[ij[0], ij[1]] += 1
bmus = np.array(bmus)

plt.figure(figsize=(6,6))
plt.title("Hit map")
plt.imshow(hits.T, origin="lower")
plt.colorbar()
plt.tight_layout()
plt.show()

# Helper to map (i,j) -> linear index on flattened weights
lin = lambda ij: ij[0] * gy + ij[1]

# -------------------------
# Model selection over K using same SOM
# -------------------------
weights = som.get_weights().reshape(-1, p)   # (gx*gy) x p
Ks = range(3, 9)
sil_vals = []
stability_ari = []

for K in Ks:
    # base kmeans and silhouette on sample labels
    km0 = KMeans(n_clusters=K, random_state=0, n_init=20).fit(weights)
    node_labels0 = km0.labels_
    sample_labels0 = np.array([node_labels0[lin(ij)] for ij in bmus])
    sil_vals.append(silhouette_score(X, sample_labels0))

    # stability across seeds
    runs = []
    for s in range(1, 6):
        kms = KMeans(n_clusters=K, random_state=s, n_init=20).fit(weights)
        node_lab = kms.labels_
        runs.append(np.array([node_lab[lin(ij)] for ij in bmus]))
    aris = [adjusted_rand_score(runs[i], runs[j])
            for i in range(len(runs)) for j in range(i+1, len(runs))]
    stability_ari.append(np.mean(aris))

# Plot selection curves
plt.figure(figsize=(8,4))
plt.plot(list(Ks), sil_vals, marker='o', label='silhouette')
plt.plot(list(Ks), stability_ari, marker='s', label='stability ARI')
plt.xlabel('K')
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig(SELECTION_CURVES_FILE, dpi=300)
plt.show()
print(f"Saved selection curves plots: {SELECTION_CURVES_FILE}")

best_k_sil = list(Ks)[int(np.argmax(sil_vals))]
best_k = 4 #best_k_sil ## 4 is chosen as the largest number of clusters before significant drop in silhouette score
print(f"Best K by silhouette: {best_k}")
print("Silhouette curve:", [round(v, 3) for v in sil_vals])
print("Stability ARI:", [round(v, 3) for v in stability_ari])

# -------------------------
# Final clustering with chosen K
# -------------------------
km_final = KMeans(n_clusters=best_k, random_state=0, n_init=50).fit(weights)
node_labels = km_final.labels_
centers = km_final.cluster_centers_

# Labels for samples via BMU node
sample_labels = np.array([node_labels[lin(ij)] for ij in bmus])

# Soft probabilities for each sample to every cluster
D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # n x K
inv = 1.0 / (1.0 + D)
cluster_probs = inv / inv.sum(axis=1, keepdims=True)              # rows sum to 1

# Sizes
unique, counts = np.unique(sample_labels, return_counts=True)
print("Participants per cluster:", dict(zip(unique, counts)))

# Component planes to see what drives separation
W = som.get_weights()  # gx x gy x p
fig, axs = plt.subplots(1, len(BEHAVIORAL_VARS), figsize=(4*len(BEHAVIORAL_VARS), 4))
if len(BEHAVIORAL_VARS) == 1:
    axs = [axs]
for j, var in enumerate(BEHAVIORAL_VARS):
    plane = W[:, :, j].T
    im = axs[j].imshow(plane, origin="lower")
    axs[j].set_title(var)
    plt.colorbar(im, ax=axs[j], fraction=.046, pad=.04)
plt.tight_layout()
plt.show()

# SOM colored by K
plt.figure(figsize=(6,6))
plt.title(f"SOM map colored by K={best_k}")
som_map = node_labels.reshape(gx, gy).T
plt.imshow(som_map, origin="lower", cmap='tab10')
plt.colorbar()
plt.tight_layout()
plt.show()

# Cluster centers in original scale for interpretation
cluster_means = scaler.inverse_transform(centers)
print("\nCluster means (original scale)")
for i in range(best_k):
    print(f"  Cluster {i}:")
    for j, var in enumerate(BEHAVIORAL_VARS):
        print(f"    {var}: {cluster_means[i, j]:.2f}")

# =========================
# Radar plots per cluster
# =========================
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

labels = [pretty_labels[v] for v in BEHAVIORAL_VARS]
p = len(labels)

# Option: normalize across clusters so all radars share a 0–1 scale
# Comment this block if you prefer true T-score values
vals = cluster_means.copy()                         # K x p (original scale from earlier)
vmin = vals.min(axis=0); vmax = vals.max(axis=0)
rng = np.where((vmax - vmin) == 0, 1, (vmax - vmin))
vals_norm = (vals - vmin) / rng                # 0..1 for plotting
radar_vals = vals_norm                              # change to vals use absolute scale

# Angles for the polygon
angles = np.linspace(0, 2*np.pi, p, endpoint=False)
angles = np.concatenate([angles, angles[:1]])       # close the loop

# Layout
K = vals.shape[0]
rows, cols = ceil(K/2), 2 if K > 1 else 1
fig = plt.figure(figsize=(5*cols, 5*rows))

# Map cluster id -> size
sizes = dict(zip(unique, counts))

for k in range(K):
    ax = plt.subplot(rows, cols, k+1, projection='polar')
    # values for this cluster, closed polygon
    v = radar_vals[k]
    data = np.concatenate([v, v[:1]])

    # draw outer reference polygon (pentagon outline)
    ref = np.ones(p)
    ref = np.concatenate([ref, ref[:1]])
    # ax.plot(angles, ref, color='black', linewidth=2)
    
    # cluster polygon
    ax.plot(angles, data, linewidth=2, color='purple')
    ax.fill(angles, data, alpha=0.15, color='purple')

    # formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([])                   # cleaner look
    ax.set_ylim(0, 1)                   # because we normalized 0..1
    # min_T = vals.min()
    # max_T = vals.max()
    # ax.set_ylim(min_T, max_T)
    ax.set_title(f"Cluster {k}  (n={sizes.get(k,0)})", fontsize=13, pad=14)

plt.tight_layout()
out_png = RADAR_PLOTS_FILE
plt.savefig(out_png, dpi=300)
plt.show()
print(f"Saved radar plots: {out_png}")


# -------------------------
# Save results
# -------------------------
df_out = df.copy()
df_out['cluster'] = sample_labels
for k in range(best_k):
    df_out[f'cluster_{k}_prob'] = cluster_probs[:, k]

out_path = OUTPUT_FILE
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_out.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")

# Final report
print(f"\nSummary")
print(f"  SOM grid: {gx}x{gy}  neurons={gx*gy}")
print(f"  QE: {qe:.4f}  TE: {te:.4f}")
print(f"  Chosen K: {best_k}")
print(f"  Silhouette at K: {silhouette_score(X, sample_labels):.4f}")
