import streamlit as st
import os
import pandas as pd
import sqlite3
from utils.database import init_db, get_datasets, get_clustering_k_values
from utils.visualization import plot_pca_scatter
from utils.ui import setup_page


def main():
    st.set_page_config(page_title="Visualization", layout="wide")
    setup_page()
    st.title("Visualisasi PCA 2D")

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
    except Exception:
        st.error(f"Tabel clustering_{dataset_id}_k{k} tidak ditemukan. Jalankan ulang clustering dengan K={k}.")
        conn.close()
        return
    conn.close()

    if 'cluster' not in df.columns:
        st.error("Data clustering tidak memiliki kolom 'cluster'.")
        return

    X = df.drop(columns=['cluster'])
    labels = df['cluster'].values

    fig = plot_pca_scatter(X, labels, None)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
