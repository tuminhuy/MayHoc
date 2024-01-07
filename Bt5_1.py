import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#tao du lieu mau
n_samples = 300
n_features = 2
n_clusters = 3

X, _= make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

#khoi tao va huan luyen mo hinh K-Means
KMeans = KMeans(n_clusters=n_clusters, random_state=42)
KMeans.fit(X)

# du doan nhan cua cac diem du lieu
labels = KMeans.predict(X)

#lay tao do cua cac trung tam cum
cluster_centers = KMeans.cluster_centers_

# ve bieu do ket qua
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='X', label='trung tam cum')
plt.legend()
plt.title('K-Means Clustering')
plt.show()
