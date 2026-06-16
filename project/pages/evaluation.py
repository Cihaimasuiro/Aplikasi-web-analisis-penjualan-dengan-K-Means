import streamlit as st
import os
from utils.database import init_db, get_last_clustering, get_clustering_history, get_clustering_result
from utils.evaluation import interpret_scores, build_evaluation_pdf
from utils.ui import setup_page


def main():
    st.set_page_config(page_title="Evaluation", layout="wide")
    setup_page()
    st.title("Evaluasi Cluster")

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    history = get_clustering_history(db_path)
    if history is None or len(history) == 0:
        st.info("Belum ada hasil clustering. Jalankan clustering terlebih dahulu.")
        return

    selected_id = st.selectbox("Pilih hasil clustering", history["id"].tolist())
    selected = get_clustering_result(db_path, int(selected_id))
    last = selected or get_last_clustering(db_path)

    st.subheader("Hasil Terakhir")
    st.json(last)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Davies-Bouldin", round(last['davies_bouldin_index'] or 0, 4))
    with col2:
        st.metric("Calinski-Harabasz", round(last['calinski_harabasz_score'] or 0, 4))

    st.subheader("Interpretasi Otomatis")
    interp = interpret_scores({
        'silhouette': last['silhouette_score'],
        'davies_bouldin': last['davies_bouldin_index'],
        'calinski_harabasz': last['calinski_harabasz_score']
    })
    for k, v in interp.items():
        st.write(f"**{k}**: {v}")

    pdf_bytes = build_evaluation_pdf(last, interp)
    st.download_button(
        "Download Evaluasi (PDF)",
        data=pdf_bytes,
        file_name=f"evaluasi_clustering_{last['id']}.pdf",
        mime="application/pdf",
    )

    st.markdown("---")
    st.subheader("Riwayat Evaluasi")
    st.dataframe(history, use_container_width=True)


if __name__ == '__main__':
    main()
