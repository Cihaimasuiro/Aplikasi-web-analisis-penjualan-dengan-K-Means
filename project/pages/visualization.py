import streamlit as st
import os
import pandas as pd
import sqlite3
import numpy as np
from utils.database import init_db, get_datasets, get_clustering_k_values
from utils.visualization import plot_pca_scatter, plot_pca_3d, plot_scatter_2d, plot_scatter_3d
from utils.ui import setup_page


def main():
    st.set_page_config(page_title="Visualization", layout="wide")
    setup_page()
    st.title("Visualisasi Klaster")

    visualization_type = st.selectbox(
        "Pilih Jenis Visualisasi",
        ["PCA 2D", "PCA 3D", "Scatter Plot 2D", "Scatter Plot 3D"]
    )

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    datasets = get_datasets(db_path)
    if datasets is None or len(datasets) == 0:
        st.info("Belum ada dataset. Upload terlebih dahulu pada halaman Upload Dataset.")
        return

    dataset_id = st.selectbox("Pilih dataset", datasets['id'].tolist())

    k_values = get_clustering_k_values(db_path, int(dataset_id))
    if not k_values:
        st.error("Belum ada hasil clustering untuk dataset ini. Jalankan clustering terlebih dahulu.")
        return

    k = st.selectbox("Pilih jumlah cluster (K)", k_values, index=len(k_values) - 1)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM clustering_{dataset_id}_k{k} LIMIT 1000",
            conn,
        )
    except Exception as e:
        st.error(f"Tabel clustering_{dataset_id}_k{k} tidak ditemukan. Jalankan ulang clustering dengan K={k}. Error: {e}")
        conn.close()
        return
    conn.close()

    if 'cluster' not in df.columns:
        st.error("Data clustering tidak memiliki kolom 'cluster'.")
        return

    X = df.drop(columns=['cluster'])
    labels = df['cluster'].values

    # Select only numeric columns for PCA and scatter plots
    X_numeric = X.select_dtypes(include=np.number)

    if X_numeric.empty:
        st.error("Tidak ada kolom numerik yang tersedia untuk visualisasi.")
        return

    if visualization_type == "PCA 2D":
        if X_numeric.shape[1] < 2:
            st.warning("Diperlukan setidaknya 2 fitur numerik untuk PCA 2D.")
        else:
            fig = plot_pca_scatter(X_numeric, labels, title=f"PCA 2D - Dataset {dataset_id} (K={k})")
            st.plotly_chart(fig, use_container_width=True)
    elif visualization_type == "PCA 3D":
        if X_numeric.shape[1] < 3:
            st.warning("Diperlukan setidaknya 3 fitur numerik untuk PCA 3D.")
        else:
            fig = plot_pca_3d(X_numeric, labels, title=f"PCA 3D - Dataset {dataset_id} (K={k})")
            st.plotly_chart(fig, use_container_width=True)
    elif visualization_type == "Scatter Plot 2D":
        features = X_numeric.columns.tolist()
        if len(features) < 2:
            st.warning("Diperlukan setidaknya 2 fitur numerik untuk Scatter Plot 2D.")
        else:
            col_x, col_y = st.columns(2)
            feature_x = col_x.selectbox("Pilih Fitur X", features, index=0)
            feature_y = col_y.selectbox("Pilih Fitur Y", [f for f in features if f != feature_x], index=0 if features[0] != feature_x else (1 if len(features) > 1 else 0))

            if feature_x and feature_y:
                fig = plot_scatter_2d(X_numeric, feature_x, feature_y, labels, title=f"Scatter Plot 2D: {feature_x} vs {feature_y} (K={k})")
                st.plotly_chart(fig, use_container_width=True)
    elif visualization_type == "Scatter Plot 3D":
        features = X_numeric.columns.tolist()
        if len(features) < 3:
            st.warning("Diperlukan setidaknya 3 fitur numerik untuk Scatter Plot 3D.")
        else:
            col_x, col_y, col_z = st.columns(3)
            feature_x = col_x.selectbox("Pilih Fitur X", features, index=0)
            remaining_features_y = [f for f in features if f != feature_x]
            feature_y = col_y.selectbox("Pilih Fitur Y", remaining_features_y, index=0 if len(remaining_features_y) > 0 else 0)
            remaining_features_z = [f for f in remaining_features_y if f != feature_y]
            feature_z = col_z.selectbox("Pilih Fitur Z", remaining_features_z, index=0 if len(remaining_features_z) > 0 else 0)

            if feature_x and feature_y and feature_z:
                fig = plot_scatter_3d(X_numeric, feature_x, feature_y, feature_z, labels, title=f"Scatter Plot 3D: {feature_x} vs {feature_y} vs {feature_z} (K={k})")
                st.plotly_chart(fig, use_container_width=True)



if __name__ == '__main__':
    main()
