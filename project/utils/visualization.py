import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


# Elbow plot using WCSS values
def plot_elbow(wcss: list, k_range: range):
    df = pd.DataFrame({"k": list(k_range), "wcss": wcss})
    fig = px.line(df, x="k", y="wcss", markers=True, title="Elbow Method (WCSS)")
    fig.update_layout(xaxis=dict(dtick=1))
    return fig


# PCA 2D scatter with centroids
def plot_pca_scatter(X, labels, centroids=None):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df = pd.DataFrame(coords, columns=["pc1", "pc2"])
    df["cluster"] = labels.astype(str)
    fig = px.scatter(df, x="pc1", y="pc2", color="cluster", title="PCA 2D - Clusters", hover_data=df.columns.tolist())
    if centroids is not None:
        cent_coords = pca.transform(centroids)
        fig.add_trace(go.Scatter(x=cent_coords[:, 0], y=cent_coords[:, 1], mode='markers', marker=dict(symbol='x', size=12, color='black'), name='centroids'))
    fig.update_layout(legend_title_text='Cluster')
    return fig