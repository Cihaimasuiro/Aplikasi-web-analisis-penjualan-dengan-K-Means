import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple, List


# Compute WCSS for range of k values
def compute_wcss(X: pd.DataFrame, k_range: range) -> List[float]:
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss


# Run kmeans and return labels and centroids
def run_kmeans(X: pd.DataFrame, k: int) -> Tuple[np.ndarray, np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids, kmeans
