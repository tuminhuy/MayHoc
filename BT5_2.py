import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

#load tap du lieu Iris
iris = datasets.load_iris()
X = iris.data # lay cac dac trung cua du lieu

#chon so cum ma ban muon phan loai du lieu vao
n_clusters = 3

#tao mo hinh k-means
KMeans = KMeans(n_clusters=n_clusters)
KMeans.fit(X) #huan luyen mo hinh tren du lieu

#lay cac trung tam cua cac cum va du doan cum cho moi diem du lieu
clusters_centers= KMeans.cluster_centers_
labels = KMeans.labels_

#hien tu ket qua clustering
plt.figure(figsize=(12, 6))

#ve du lieu theo tung cum
colors = ['red', 'green', 'pink']
for i in range(n_clusters):
    plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], c=colors[i], label=f'Cluster {i +1}')

#ve trung tam cua cac cum
    plt.scatter(clusters_centers[:, 0], clusters_centers[:, 1], c='black', marker='X', s=200, label='Cluster Centers')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.legend()
    plt.show()
