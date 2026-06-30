"""
Halaman K-Means Clustering — mendukung dua mode:
  1. Mode RFM   : deteksi otomatis kolom CustomerID / Tanggal / Monetary
  2. Mode Manual: pilih fitur bebas dari tabel yang sudah ter-preprocess

Alur Otomasi K:
  • Tekan "Hitung Rekomendasi K" → sistem menjalankan Elbow + Silhouette
  • Nilai K terbaik (konsensus) langsung mengisi input K di bagian bawah
  • Pengguna tetap bisa mengubah K secara manual sebelum menjalankan K-Means
"""

import os
import sqlite3

import pandas as pd
import streamlit as st
import numpy as np

from utils.clustering import compute_wcss, detect_elbow_k, recommend_k, run_kmeans
from utils.database import (
    get_best_k_recommendations,
    get_clustering_history,
    get_dataset_table,
    get_datasets,
    init_db,
    save_clustering_result,
    save_selected_features,
)
from utils.evaluation import compute_scores, interpret_scores
from utils.rfm import compute_rfm, detect_rfm_columns, rfm_summary_table, scale_rfm
from utils.ui import setup_page
from utils.visualization import plot_elbow


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _silhouette_badge(score: float) -> str:
    if score > 0.6:
        return "🟢 Sangat Baik"
    if score > 0.4:
        return "🟡 Baik / Sedang"
    return "🔴 Lemah"


def _cluster_label(summary_row: pd.Series, k: int) -> str:
    rank = int(summary_row["Ranking Nilai Pelanggan"])
    labels = [
        "Champions 🏆",
        "Loyal Customers ⭐",
        "Regular Customers 📦",
        "Potential Loyalists 🌱",
        "At Risk ⚠️",
        "Needs Attention 🔔",
        "Lost Customers 💤",
        "New Customers 🆕",
    ]
    return labels[rank - 1] if rank <= len(labels) else f"Segmen {rank}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Clustering", layout="wide")
    setup_page()
    st.title("K-Means Clustering")
    st.caption("Segmentasi pelanggan menggunakan RFM + K-Means dengan rekomendasi K otomatis.")

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    datasets = get_datasets(db_path)
    if datasets is None or len(datasets) == 0:
        st.info("Belum ada dataset. Upload terlebih dahulu pada halaman Upload Dataset.")
        return

    dataset_id = st.selectbox("Pilih dataset", datasets["id"].tolist())

    # Load preprocessed if exists, else raw
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM 'preprocessed_{dataset_id}'", conn)
    except Exception:
        df = get_dataset_table(db_path, int(dataset_id))
    conn.close()

    # ── Mode selector ────────────────────────────────────────────────────────
    mode = st.radio(
        "Mode Clustering",
        ["🔵 Mode RFM (Otomatis)", "⚙️ Mode Manual (Pilih Fitur)"],
        horizontal=True,
    )

    rfm_df = None
    X = None
    rfm_mode = mode.startswith("🔵")

    st.markdown("---")

    # ── RFM Mode ─────────────────────────────────────────────────────────────
    if rfm_mode:
        st.subheader("Konfigurasi RFM")
        detected = detect_rfm_columns(df)

        col_a, col_b, col_c = st.columns(3)
        customer_col = col_a.selectbox(
            "Kolom Customer ID *",
            df.columns.tolist(),
            index=df.columns.tolist().index(detected["customer"]) if detected["customer"] else 0,
        )
        date_col = col_b.selectbox(
            "Kolom Tanggal *",
            df.columns.tolist(),
            index=df.columns.tolist().index(detected["date"]) if detected["date"] else 0,
        )

        monetary_options = ["(tidak ada / hitung dari Qty × Harga)"] + df.columns.tolist()
        monetary_col_choice = col_c.selectbox(
            "Kolom Total Nilai (Monetary)",
            monetary_options,
            index=(monetary_options.index(detected["monetary"]) if detected["monetary"] and detected["monetary"] in monetary_options else 0),
        )
        monetary_col = None if monetary_col_choice.startswith("(") else monetary_col_choice

        col_d, col_e, col_f = st.columns(3)
        qty_col = col_d.selectbox(
            "Kolom Qty (jika tidak ada Total Nilai)",
            ["(tidak dipakai)"] + df.columns.tolist(),
            index=0,
        )
        price_col = col_e.selectbox(
            "Kolom Harga Satuan (jika tidak ada Total Nilai)",
            ["(tidak dipakai)"] + df.columns.tolist(),
            index=0,
        )
        invoice_col = col_f.selectbox(
            "Kolom Invoice (untuk Frequency — opsional)",
            ["(tidak dipakai)"] + df.columns.tolist(),
            index=0,
        )

        qty_col = None if qty_col.startswith("(") else qty_col
        price_col = None if price_col.startswith("(") else price_col
        invoice_col = None if invoice_col.startswith("(") else invoice_col

        if st.button("🔢 Hitung Nilai RFM"):
            try:
                with st.spinner("Menghitung nilai RFM …"):
                    rfm_df, diag = compute_rfm(
                        df,
                        customer_col=customer_col,
                        date_col=date_col,
                        monetary_col=monetary_col,
                        quantity_col=qty_col,
                        price_col=price_col,
                        invoice_col=invoice_col,
                    )
                st.session_state["rfm_df"] = rfm_df
                st.session_state["rfm_diag"] = diag
                st.success(f"RFM berhasil dihitung untuk {len(rfm_df):,} pelanggan.")
            except Exception as e:
                st.error(f"Gagal menghitung RFM: {e}")

        if "rfm_diag" in st.session_state:
            diag = st.session_state["rfm_diag"]
            with st.expander("🔬 Diagnostik Proses RFM — klik untuk melihat kenapa jumlah pelanggan berubah", expanded=("rfm_df" in st.session_state and len(st.session_state["rfm_df"]) < st.session_state["rfm_diag"].get("step1_unique_customers", 9999))):
                st.markdown("#### Step 1 — Validasi Customer")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total baris dataset", f"{diag['step1_total_rows']:,}")
                c2.metric("Unique customer (raw)", f"{diag['step1_unique_customers']:,}")
                c3.metric("Baris tanpa Customer ID (dibuang)", f"{diag['step1_rows_dropped_no_customer']:,}")
                st.caption(f"Kolom customer yang dipakai: **{diag['step1_customer_col']}** | Contoh nilai: `{diag['step1_sample_customers']}`")

                st.markdown("#### Step 2 — Parsing Tanggal")
                c1, c2, c3 = st.columns(3)
                c1.metric("Baris gagal parse tanggal (dibuang)", f"{diag['step2_rows_failed_date_parse']:,}", delta=f"-{diag['step2_rows_failed_date_parse']:,}" if diag['step2_rows_failed_date_parse'] > 0 else None, delta_color="inverse")
                c2.metric("Baris setelah filter tanggal", f"{diag['step2_rows_after_date_drop']:,}")
                c3.metric("Reference date", diag.get('step2_reference_date', '—'))
                st.caption(f"Rentang tanggal: {diag.get('step2_date_range', '—')} | Contoh nilai tanggal raw: `{diag['step2_sample_raw_dates']}`")
                if diag['step2_rows_failed_date_parse'] > 0:
                    st.warning(
                        f"⚠️ **{diag['step2_rows_failed_date_parse']:,} baris dibuang** karena format tanggal tidak bisa di-parse. "
                        "Periksa apakah ada nilai kosong, teks non-tanggal, atau format campuran di kolom tanggal."
                    )

                st.markdown("#### Step 3 — Nilai Monetary")
                st.caption(f"Sumber monetary: **{diag['step3_monetary_source']}**")
                amt = diag['step3_amount_stats']
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Min", f"{amt['min']:,}")
                c2.metric("Max", f"{amt['max']:,}")
                c3.metric("Rata-rata", f"{amt['mean']:,}")
                c4.metric("% nilai = 0", f"{amt['zero_pct']}%")
                if diag['step3_rows_nan_amount'] > 0:
                    st.warning(f"⚠️ {diag['step3_rows_nan_amount']:,} baris punya nilai Monetary NaN → diisi 0.")

                st.markdown("#### Step 4 — Groupby & RFM Output")
                c1, c2, c3 = st.columns(3)
                c1.metric("Pelanggan sebelum filter", f"{diag['step4_customers_before_filter']:,}")
                c2.metric("Pelanggan setelah filter (Frequency > 0)", f"{diag['step4_customers_after_filter']:,}")
                c3.metric("Sumber Frequency", diag['step4_frequency_source'])
                rfm_s = diag['step4_rfm_stats']
                st.caption(
                    f"Recency: min={rfm_s['recency']['min']} hari, max={rfm_s['recency']['max']} hari, avg={rfm_s['recency']['mean']} | "
                    f"Frequency: min={rfm_s['frequency']['min']}, max={rfm_s['frequency']['max']}, avg={rfm_s['frequency']['mean']} | "
                    f"Monetary: min={rfm_s['monetary']['min']:,}, max={rfm_s['monetary']['max']:,}, avg={rfm_s['monetary']['mean']:,}"
                )

        if "rfm_df" in st.session_state:
            rfm_df = st.session_state["rfm_df"]
            st.subheader("Tabel RFM per Pelanggan")

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Avg Recency (hari)", f"{rfm_df['Recency'].mean():.1f}")
            col_stat2.metric("Avg Frequency", f"{rfm_df['Frequency'].mean():.1f}")
            col_stat3.metric("Avg Monetary (Rp)", f"{rfm_df['Monetary'].mean():,.0f}")

            st.dataframe(rfm_df.head(50), use_container_width=True)
            X = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
            # Normalize for clustering
            X = scale_rfm(rfm_df)[["Recency", "Frequency", "Monetary"]]

    # ── Manual Mode ───────────────────────────────────────────────────────────
    else:
        st.subheader("Preview data")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Pilih fitur untuk clustering")
        features = st.multiselect(
            "Fitur numerik",
            df.columns.tolist(),
            default=df.select_dtypes(include=["number"]).columns.tolist(),
        )
        if st.button("Simpan Fitur Terpilih") and features:
            save_selected_features(db_path, int(dataset_id), features)
            st.success("Fitur tersimpan.")

        if features:
            X = df[features].dropna()
            st.info(f"{len(X):,} baris tersedia untuk clustering.")

    # ── Elbow + Rekomendasi K ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 Elbow Method & Rekomendasi K")

    if X is None or len(X) == 0:
        st.info("Hitung RFM atau pilih fitur terlebih dahulu.")
        return

    # Ensure X contains only numeric columns before passing to clustering algorithms
    X_numeric = X.select_dtypes(include=np.number)

    if len(X_numeric.columns) == 0:
        st.error("Tidak ada fitur numerik yang tersedia untuk clustering. Harap pilih fitur numerik.")
        return

    max_k = min(10, len(X_numeric) - 1)
    k_range = range(1, max_k + 1)

    if st.button("🔍 Hitung Rekomendasi K (Elbow + Silhouette)"):
        with st.spinner("Menghitung WCSS dan Silhouette untuk k = 1 … " + str(max_k) + " …"):
            wcss = compute_wcss(X_numeric, k_range)
            rec = recommend_k(wcss, k_range, X_numeric)
            st.session_state["wcss"] = wcss
            st.session_state["rec"] = rec

    if "wcss" in st.session_state and "rec" in st.session_state:
        wcss = st.session_state["wcss"]
        rec = st.session_state["rec"]

        fig = plot_elbow(wcss, k_range)
        st.plotly_chart(fig, use_container_width=True)

        # Silhouette table
        if rec["silhouette_scores"]:
            sil_df = pd.DataFrame(rec["silhouette_scores"])
            sil_df["Interpretasi"] = sil_df["silhouette"].apply(_silhouette_badge)
            st.subheader("Silhouette Score per Nilai K")
            st.dataframe(sil_df, use_container_width=True)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("📍 K Rekomendasi Elbow", rec["elbow_k"])
        col_r2.metric(
            "📈 K Rekomendasi Silhouette",
            rec["silhouette_k"] if rec["silhouette_k"] else "—",
        )
        col_r3.metric("✅ K Konsensus (default)", rec["final_k"])

        # Auto-fill session state K
        st.session_state["recommended_k"] = rec["final_k"]
        st.success(
            f"Sistem merekomendasikan **K = {rec['final_k']}**. "
            "Nilai ini sudah diisi otomatis di bawah, namun bisa kamu ubah secara manual."
        )

    # ── Jalankan K-Means ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚀 Jalankan K-Means")

    default_k = st.session_state.get("recommended_k", min(3, max_k))
    k = st.number_input(
        "Jumlah Cluster (K)",
        min_value=2,
        max_value=max(2, max_k),
        value=int(default_k),
        step=1,
        help="Nilai K diisi otomatis dari rekomendasi di atas, tapi bisa diubah manual.",
    )

    if st.button("▶️ Jalankan K-Means"):
        with st.spinner(f"Menjalankan K-Means dengan K = {k} …"):
            labels, centroids, model = run_kmeans(X_numeric, int(k))
            scores = compute_scores(X, labels)

            # Save to DB
            conn = sqlite3.connect(db_path)
            table_name = f"clustering_{dataset_id}_k{k}"
            df_result = X.copy()
            df_result["cluster"] = labels
            if rfm_mode and rfm_df is not None:
                # Merge customer ID back
                df_full = rfm_df.copy()
                df_full["cluster"] = labels
                df_full.to_sql(table_name, conn, if_exists="replace", index=False)
            else:
                df_result.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()

            res_id = save_clustering_result(
                db_path,
                int(dataset_id),
                int(k),
                scores.get("silhouette"),
                scores.get("davies_bouldin"),
                scores.get("calinski_harabasz"),
            )

        st.success("Clustering selesai ✅")
        st.session_state["cluster_labels"] = labels
        st.session_state["cluster_centroids"] = centroids
        st.session_state["cluster_scores"] = scores
        st.session_state["cluster_k"] = int(k)
        if rfm_mode and rfm_df is not None:
            st.session_state["cluster_rfm_df"] = rfm_df

    # ── Hasil & Evaluasi ──────────────────────────────────────────────────────
    if "cluster_labels" in st.session_state:
        labels     = st.session_state["cluster_labels"]
        centroids  = st.session_state["cluster_centroids"]
        scores     = st.session_state["cluster_scores"]
        k_used     = st.session_state["cluster_k"]
        rfm_result = st.session_state.get("cluster_rfm_df")

        st.markdown("---")
        st.subheader("📊 Evaluasi Model")

        sil  = scores.get("silhouette", 0) or 0
        db_s = scores.get("davies_bouldin", 0) or 0
        ch_s = scores.get("calinski_harabasz", 0) or 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{sil:.4f}", _silhouette_badge(sil))
        c2.metric("Davies-Bouldin Index", f"{db_s:.4f}", "↓ semakin kecil semakin baik")
        c3.metric("Calinski-Harabasz Score", f"{ch_s:.1f}", "↑ semakin besar semakin baik")

        interp = interpret_scores(scores)
        with st.expander("📖 Interpretasi Metrik Evaluasi"):
            for key, val in interp.items():
                st.write(f"**{key}**: {val}")

        # ── Tabel Centroid ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📍 Nilai Centroid Tiap Cluster")
        feat_names = (
            ["Recency (norm)", "Frequency (norm)", "Monetary (norm)"]
            if rfm_mode else list(X.columns)
        )
        centroid_df = pd.DataFrame(
            centroids,
            columns=feat_names[:len(centroids[0])],
        )
        centroid_df.insert(0, "Cluster", range(len(centroids)))
        st.dataframe(centroid_df.style.format({c: "{:.4f}" for c in feat_names[:len(centroids[0])]}), use_container_width=True)

        # ── RFM Cluster Summary ───────────────────────────────────────────────
        if rfm_mode and rfm_result is not None:
            st.markdown("---")
            st.subheader("🗂️ Profil RFM Tiap Cluster")

            summary = rfm_summary_table(rfm_result, labels)

            # Assign human-readable segment label
            summary["Segmen"] = summary.apply(lambda row: _cluster_label(row, k_used), axis=1)
            summary = summary[[
                "Cluster", "Segmen", "Jumlah Pelanggan", "% Pelanggan",
                "Avg Recency (hari)", "Avg Frequency", "Avg Monetary (Rp)",
                "Total Monetary (Rp)", "Ranking Nilai Pelanggan",
            ]]

            st.dataframe(
                summary.style.format({
                    "% Pelanggan": "{:.1f}%",
                    "Avg Recency (hari)": "{:.1f}",
                    "Avg Frequency": "{:.1f}",
                    "Avg Monetary (Rp)": "Rp {:,.0f}",
                    "Total Monetary (Rp)": "Rp {:,.0f}",
                }),
                use_container_width=True,
            )

            # Narasi otomatis per cluster
            st.subheader("📝 Narasi Otomatis per Cluster")
            for _, row in summary.iterrows():
                with st.expander(f"Cluster {int(row['Cluster'])} — {row['Segmen']}"):
                    st.markdown(
                        f"Cluster ini memiliki **{int(row['Jumlah Pelanggan'])} pelanggan** "
                        f"({row['% Pelanggan']:.1f}% dari total), "
                        f"dengan rata-rata Recency **{row['Avg Recency (hari)']:.0f} hari** "
                        f"sejak transaksi terakhir, "
                        f"Frequency rata-rata **{row['Avg Frequency']:.0f} transaksi**, "
                        f"dan Monetary rata-rata **Rp {row['Avg Monetary (Rp)']:,.0f}** per pelanggan. "
                        f"Total kontribusi revenue cluster ini: **Rp {row['Total Monetary (Rp)']:,.0f}**. "
                        f"Cluster ini berada pada **peringkat {int(row['Ranking Nilai Pelanggan'])}** "
                        f"dari {k_used} cluster berdasarkan nilai pelanggan (Customer Value)."
                    )

            # Download full RFM + cluster
            rfm_out = rfm_result.copy()
            rfm_out["cluster"] = labels
            rfm_out = rfm_out.merge(
                summary[["Cluster", "Segmen"]],
                left_on="cluster",
                right_on="Cluster",
                how="left",
            ).drop(columns=["Cluster"])
            csv = rfm_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download RFM + Cluster (CSV)", data=csv, file_name="rfm_clustering.csv", mime="text/csv")
        else:
            # Generic result table
            df_result = X.copy()
            df_result["cluster"] = labels
            st.subheader("Contoh Hasil (50 baris)")
            st.dataframe(df_result.head(50), use_container_width=True)
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil Clustering (CSV)", data=csv, file_name="hasil_clustering.csv", mime="text/csv")

    # ── Riwayat ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📜 Riwayat Clustering Dataset Ini")
    history = get_clustering_history(db_path)
    if len(history) > 0:
        dh = history[history["dataset_id"] == int(dataset_id)]
        st.dataframe(dh, use_container_width=True)
        recs = get_best_k_recommendations(db_path, int(dataset_id))
        if recs["silhouette"] and recs["silhouette"]["k_value"] is not None and recs["silhouette"]["silhouette_score"] is not None:
            st.info(f"Terbaik (Silhouette): K = {int(recs['silhouette']['k_value'])} → score = {recs['silhouette']['silhouette_score']:.4f}")
        if recs["davies_bouldin"] and recs["davies_bouldin"]["k_value"] is not None and recs["davies_bouldin"]["davies_bouldin_index"] is not None:
            st.info(f"Terbaik (Davies-Bouldin): K = {int(recs['davies_bouldin']['k_value'])} → index = {recs['davies_bouldin']['davies_bouldin_index']:.4f}")
    else:
        st.info("Belum ada riwayat clustering.")


if __name__ == "__main__":
    main()
