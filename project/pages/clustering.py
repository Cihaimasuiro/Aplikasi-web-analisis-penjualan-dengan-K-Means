import streamlit as st
import os
import pandas as pd
from utils.database import (
    init_db,
    get_datasets,
    get_dataset_table,
    save_selected_features,
    save_clustering_result,
    get_clustering_history,
    get_best_k_recommendations,
)
from utils.clustering import compute_wcss, run_kmeans
from utils.visualization import plot_elbow
from utils.evaluation import compute_scores
from utils.ui import setup_page


def main():
    st.set_page_config(page_title="Clustering", layout="wide")
    setup_page()
    st.title("K-Means Clustering")

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    datasets = get_datasets(db_path)
    if datasets is None or len(datasets) == 0:
        st.info("Belum ada dataset. Upload terlebih dahulu pada halaman Upload Dataset.")
        return

    dataset_id = st.selectbox("Pilih dataset", datasets['id'].tolist())

    # Load preprocessed if exists else raw
    import sqlite3
    conn = sqlite3.connect(db_path)
    table_pre = f"preprocessed_{dataset_id}"
    try:
        df = pd.read_sql_query(f"SELECT * FROM '{table_pre}'", conn)
    except Exception:
        df = get_dataset_table(db_path, int(dataset_id))
    conn.close()

    st.subheader("Preview data untuk clustering")
    st.dataframe(df.head(20))

    st.subheader("Pilih fitur untuk clustering")
    features = st.multiselect("Fitur", df.columns.tolist(), default=df.select_dtypes(include=['number']).columns.tolist())

    if st.button("Simpan Fitur Terpilih"):
        if len(features) == 0:
            st.error("Pilih minimal satu fitur.")
        else:
            save_selected_features(db_path, int(dataset_id), features)
            st.success("Fitur tersimpan.")

    st.markdown("---")
    st.subheader("Elbow Method")
    try:
        X = df[features].dropna()
    except Exception:
        st.error("Fitur yang dipilih tidak valid untuk clustering.")
        return

    if X.shape[0] == 0:
        st.error("Tidak ada data setelah dropna pada fitur terpilih.")
        return
    if X.shape[0] < 2:
        st.error("Minimal diperlukan 2 baris data untuk clustering.")
        return
    if len(features) == 0:
        st.error("Pilih minimal satu fitur numerik untuk clustering.")
        return

    max_k = min(10, X.shape[0] - 1)
    k_range = range(1, max_k + 1)
    if st.button("Hitung Elbow (WCSS)"):
        with st.spinner("Menghitung WCSS..."):
            wcss = compute_wcss(X, k_range)
            fig = plot_elbow(wcss, k_range)
            st.plotly_chart(fig, use_container_width=True)
            # simple recommendation: find elbow by relative drop
            drops = [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
            if drops:
                rec_k = drops.index(max(drops)) + 2
                st.info(f"Rekomendasi jumlah cluster (rule of largest WCSS drop): K = {rec_k}")

    st.subheader("Auto Recommendation K")
    if st.button("Hitung Rekomendasi K"):
        if X.shape[0] < 3:
            st.error("Minimal diperlukan 3 baris data untuk membandingkan nilai K.")
        else:
            rows = []
            with st.spinner("Menghitung Silhouette dan Davies-Bouldin untuk beberapa nilai K..."):
                for candidate_k in range(2, max_k + 1):
                    labels, centroids, model = run_kmeans(X, candidate_k)
                    scores = compute_scores(X, labels)
                    rows.append(
                        {
                            "k": candidate_k,
                            "silhouette": scores.get("silhouette"),
                            "davies_bouldin": scores.get("davies_bouldin"),
                            "calinski_harabasz": scores.get("calinski_harabasz"),
                        }
                    )
            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, use_container_width=True)
            if not result_df.empty:
                best_s = result_df.sort_values("silhouette", ascending=False).iloc[0]
                best_d = result_df.sort_values("davies_bouldin", ascending=True).iloc[0]
                col1, col2 = st.columns(2)
                col1.metric("Silhouette tertinggi", int(best_s["k"]), round(float(best_s["silhouette"]), 4))
                col2.metric("Davies-Bouldin terendah", int(best_d["k"]), round(float(best_d["davies_bouldin"]), 4))

    st.markdown("---")
    st.subheader("Jalankan K-Means")
    k = st.number_input("Jumlah Cluster (K)", min_value=2, max_value=max(2, X.shape[0] - 1), value=min(3, max(2, X.shape[0] - 1)), step=1)
    if st.button("Jalankan K-Means"):
        with st.spinner("Menjalankan K-Means..."):
            labels, centroids, model = run_kmeans(X, int(k))
            df_result = X.copy()
            df_result['cluster'] = labels

            # Save clustering result to sqlite as clustering_{dataset_id}_k{k}
            import sqlite3
            conn = sqlite3.connect(db_path)
            table_name = f"clustering_{dataset_id}_k{k}"
            try:
                df_result.to_sql(table_name, conn, if_exists='replace', index=False)
            except Exception as e:
                st.error(f"Gagal menyimpan hasil clustering: {e}")
            finally:
                conn.close()

            # Compute metrics
            scores = compute_scores(X, labels)
            res_id = save_clustering_result(db_path, int(dataset_id), int(k), scores.get('silhouette'), scores.get('davies_bouldin'), scores.get('calinski_harabasz'))

            st.success("Clustering selesai dan disimpan ke database.")
            st.subheader("Contoh hasil (20 baris)")
            st.dataframe(df_result.head(20))
            st.subheader("Metode Evaluasi")
            st.write(scores)

            # Export CSV
            try:
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Hasil Clustering (CSV)", data=csv, file_name="hasil_clustering.csv", mime='text/csv')
            except Exception as e:
                st.error(f"Gagal menyiapkan file untuk diunduh: {e}")

    st.markdown("---")
    st.subheader("Riwayat Clustering Dataset Ini")
    history = get_clustering_history(db_path)
    if len(history) > 0:
        dataset_history = history[history["dataset_id"] == int(dataset_id)]
        st.dataframe(dataset_history, use_container_width=True)
        recommendations = get_best_k_recommendations(db_path, int(dataset_id))
        if recommendations["silhouette"]:
            st.info(f"Riwayat terbaik berdasarkan Silhouette: K = {int(recommendations['silhouette']['k_value'])}")
        if recommendations["davies_bouldin"]:
            st.info(f"Riwayat terbaik berdasarkan Davies-Bouldin: K = {int(recommendations['davies_bouldin']['k_value'])}")
    else:
        st.info("Belum ada riwayat clustering.")


if __name__ == '__main__':
    main()
