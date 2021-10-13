import numpy as np
from sklearn.cluster import KMeans
from isosplit5 import isosplit5

class OverClustering:
    def __init__(self, X: np.ndarray, max_radius: float) -> None:
        self._X = X
        self._N = X.shape[0]
        self._K = X.shape[1]
        self._max_radius = max_radius
        inds = list(range(self._N))
        self._clusters = [{
            'indices': inds,
            'radius': _compute_cluster_radius(self._X[inds])
        }]
    @property
    def num_clusters(self):
        return len(self._clusters)
    def get_cluster_indices(self, i) -> np.array:
        return self._clusters[i]['indices']
    def split_cluster(self, i):
        inds = self.get_cluster_indices(i)
        X0 = self._X[inds]
        labels = isosplit5(X0.T)
        if np.max(labels) > 1:
            labels = labels - 1
        else:
            kmeans = KMeans(n_clusters=2).fit(X0)
            labels = kmeans.labels_
        inds1 = [inds[i] for i in range(len(labels)) if labels[i] == 0]
        R1 = _compute_cluster_radius(self._X[inds1])
        self._clusters[i] = {
            'indices': inds1,
            'radius': R1
        }
        for jj in range(1, np.max(labels) + 1):
            inds2 = [inds[i] for i in range(len(labels)) if labels[i] == jj]
            R2 = _compute_cluster_radius(self._X[inds2])
            self._clusters.append({
                'indices': inds2,
                'radius': R2
            })
    def split_next_cluster(self):
        for i in range(len(self._clusters)):
            if self._clusters[i]['radius'] > self._max_radius:
                self.split_cluster(i)
                return True
        return False

def _compute_cluster_radius(X: np.ndarray):
    centroid = np.mean(X, axis=0)
    dists = np.sqrt(np.sum((X - centroid) ** 2, axis=0))
    return np.max(dists)

def overcluster(X: np.ndarray, max_radius: float):
    OC = OverClustering(X, max_radius=max_radius)
    while OC.split_next_cluster():
        pass
    labels = np.zeros((X.shape[0]), dtype=np.int32)
    for i in range(OC.num_clusters):
        inds = OC.get_cluster_indices(i)
        labels[inds] = i + 1
    return labels