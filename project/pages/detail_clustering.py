import os

import streamlit as st

from utils.database import init_db, get_clustering_history, get_clustering_result, get_clustering_table
from utils.evaluation import interpret_scores
from utils.ui import setup_page


def main():
    st.set_page_config(page_title="Detail Hasil Clustering", layout="wide")
    setup_page()
    st.title("Detail Hasil Clustering")

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    history = get_clustering_history(db_path)
    if history is None or len(history) == 0:
        st.info("Belum ada hasil clustering.")
        return

    selected_id = st.selectbox("Pilih ID hasil clustering", history["id"].tolist())
    result = get_clustering_result(db_path, int(selected_id))
    if result is None:
        st.error("Hasil clustering tidak ditemukan.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", result.get("filename") or result["dataset_id"])
    col2.metric("K", result["k_value"])
    col3.metric("Davies-Bouldin", round(result["davies_bouldin_index"] or 0, 4))

    st.subheader("Metadata")
    st.json(result)

    scores = {
        "silhouette": result["silhouette_score"],
        "davies_bouldin": result["davies_bouldin_index"],
        "calinski_harabasz": result["calinski_harabasz_score"],
    }
    st.subheader("Interpretasi")
    for key, value in interpret_scores(scores).items():
        st.write(f"**{key}**: {value}")

    st.subheader("Data Hasil Clustering")
    clustered = get_clustering_table(db_path, int(result["dataset_id"]), int(result["k_value"]), limit=1000)
    if clustered.empty:
        st.warning("Tabel hasil clustering tidak ditemukan.")
        return

    st.dataframe(clustered, use_container_width=True)
    csv = clustered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Detail (CSV)",
        data=csv,
        file_name=f"detail_clustering_{result['id']}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
