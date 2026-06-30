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


def plot_pca_scatter(X, labels, centroids=None, title="PCA 2D - Clusters"):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df["cluster"] = labels.astype(str)

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        title=title,
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )

    if centroids is not None:
        try:
            cent_coords = pca.transform(centroids)
            fig.add_trace(
                go.Scatter(
                    x=cent_coords[:, 0],
                    y=cent_coords[:, 1],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="black"),
                    name="Centroids",
                )
            )
        except Exception:
            # Silently ignore centroid plotting issues if transformation fails
            pass

    fig.update_layout(
        legend_title_text="Cluster",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        title={"x": 0.5, "xanchor": "center"},
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
    return fig

def plot_scatter_3d(X, feature_x, feature_y, feature_z, labels, title="Scatter Plot 3D - Clusters"):
    df = X.copy()
    df["cluster"] = labels.astype(str)
    fig = px.scatter_3d(
        df,
        x=feature_x,
        y=feature_y,
        z=feature_z,
        color="cluster",
        title=title,
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        legend_title_text="Cluster",
        scene=dict(
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            zaxis_title=feature_z,
        ),
        title={"x": 0.5, "xanchor": "center"},
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")))
    return fig

def plot_pca_3d(X, labels, centroids=None, title="PCA 3D - Clusters"):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(X)
    df = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    df["cluster"] = labels.astype(str)

    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="cluster",
        title=title,
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )

    if centroids is not None:
        try:
            cent_coords = pca.transform(centroids)
            fig.add_trace(
                go.Scatter3d(
                    x=cent_coords[:, 0],
                    y=cent_coords[:, 1],
                    z=cent_coords[:, 2],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="black"),
                    name="Centroids",
                )
            )
        except Exception:
            pass

    fig.update_layout(
        legend_title_text="Cluster",
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3",
        ),
        title={"x": 0.5, "xanchor": "center"},
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")))
    return fig

def plot_scatter_2d(X, feature_x, feature_y, labels, title="Scatter Plot 2D - Clusters"):
    df = X.copy()
    df["cluster"] = labels.astype(str)
    fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        color="cluster",
        title=title,
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        legend_title_text="Cluster",
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        title={"x": 0.5, "xanchor": "center"},
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
    return fig

def plot_scatter_3d(X, feature_x, feature_y, feature_z, labels, title="Scatter Plot 3D - Clusters"):
    df = X.copy()
    df["cluster"] = labels.astype(str)
    fig = px.scatter_3d(
        df,
        x=feature_x,
        y=feature_y,
        z=feature_z,
        color="cluster",
        title=title,
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        legend_title_text="Cluster",
        scene=dict(
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            zaxis_title=feature_z,
        ),
        title={"x": 0.5, "xanchor": "center"},
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")))
    return fig