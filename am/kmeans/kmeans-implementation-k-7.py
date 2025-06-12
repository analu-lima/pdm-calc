import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

def tanimoto_coefficient(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / union if union != 0 else 0

wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets

print(X.columns)

X_vis = X[['Alcohol', 'Malicacid']].values

le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

k = 7
n_samples = X_vis.shape[0]
random_indices = np.random.choice(n_samples, k, replace=False)
centroids = X_vis[random_indices]
max_iters = 100

for i in range(max_iters):
    distances = np.linalg.norm(X_vis[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    new_centroids = np.array([X_vis[cluster_labels == j].mean(axis=0) for j in range(k)])
    
    if np.all(np.abs(new_centroids - centroids) < 1e-4):
        break
    centroids = new_centroids

for i in range(k):
    plt.scatter(X_vis[cluster_labels == i, 0], X_vis[cluster_labels == i, 1], label=f'Cluster {i + 1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroides')
plt.xlabel('Alcohol')
plt.ylabel('Malicacid')
plt.title('Clusters Formados (Manual)')
plt.legend()
plt.show()

cluster_one_hot = np.eye(k)[cluster_labels]
true_one_hot = np.eye(len(np.unique(y_encoded)))[y_encoded]

min_cols = min(cluster_one_hot.shape[1], true_one_hot.shape[1])
cluster_one_hot = cluster_one_hot[:, :min_cols]
true_one_hot = true_one_hot[:, :min_cols]

tanimoto_scores = [tanimoto_coefficient(cluster_one_hot[i], true_one_hot[i]) for i in range(len(X_vis))]
media_tanimoto = np.mean(tanimoto_scores)

print(f"Média da Similaridade de Tanimoto (k = {k}): {media_tanimoto:.4f}")