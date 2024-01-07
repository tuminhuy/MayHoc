import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances

#load du lieu iris
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# tinh ma tran khoang cach
distance_matrix = pairwise_distances(X)

# thuc hien hierachical clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='enclidean')
agg_clustering.fit(X)

#hien thi ket qua clustering
labels = agg_clustering.labels_
unique_labels = np.unique(labels)

#ve dendrogram
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title("Dendrogram")
plt.show()

#hien thi kq clustering
for label in unique_labels:
    cluster = X[labels == label]
    print(f"Cluster {label}:")
    print(cluster)