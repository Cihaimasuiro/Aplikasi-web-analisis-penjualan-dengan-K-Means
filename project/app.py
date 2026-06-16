import streamlit as st
from utils.database import init_db, get_datasets, get_last_clustering, get_clustering_history, get_best_k_recommendations
from utils.ui import setup_page
import os


def main():
    st.set_page_config(page_title="Customer Segmentation", layout="wide")
    setup_page()

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    st.title("Dashboard Analitik Segmentasi Pelanggan")
    st.caption("Monitoring dataset, clustering, evaluasi, dan rekomendasi jumlah cluster.")
    datasets = get_datasets(db_path)
    history = get_clustering_history(db_path, limit=8)
    last = get_last_clustering(db_path)
    recommendations = get_best_k_recommendations(db_path)

    total_datasets = 0 if datasets is None else len(datasets)
    total_rows = int(datasets["rows"].sum()) if datasets is not None and len(datasets) > 0 else 0
    total_runs = len(history)
    last_k = last["k_value"] if last is not None else "-"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", total_datasets)
    col2.metric("Total Baris", total_rows)
    col3.metric("Riwayat Clustering", total_runs)
    col4.metric("K Terakhir", last_k)

    st.markdown("---")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.subheader("Rekomendasi Silhouette")
        best = recommendations["silhouette"]
        if best:
            st.metric("K terbaik", int(best["k_value"]), round(best["silhouette_score"], 4))
        else:
            st.info("Belum ada hasil clustering.")
    with rec_col2:
        st.subheader("Rekomendasi Davies-Bouldin")
        best = recommendations["davies_bouldin"]
        if best:
            st.metric("K terbaik", int(best["k_value"]), round(best["davies_bouldin_index"], 4))
        else:
            st.info("Belum ada hasil clustering.")

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Dataset Terdaftar")
        if datasets is not None and len(datasets) > 0:
            st.dataframe(datasets, use_container_width=True)
        else:
            st.info("Belum ada dataset yang terupload.")
    with right:
        st.subheader("Riwayat Clustering Terbaru")
        if history is not None and len(history) > 0:
            st.dataframe(history, use_container_width=True)
        else:
            st.info("Belum ada riwayat clustering.")

    if last is not None:
        st.subheader("Hasil Clustering Terakhir")
        st.json(last)


if __name__ == "__main__":
    main()
