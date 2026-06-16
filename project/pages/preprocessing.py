import streamlit as st
import os
import pandas as pd
from utils.database import init_db, get_datasets, get_dataset_table
from utils.preprocessing import dataset_info, drop_missing, label_encode, standard_scale
from utils.ui import setup_page


def main():
    st.set_page_config(page_title="Preprocessing", layout="wide")
    setup_page()
    st.title("Preprocessing Data")

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    datasets = get_datasets(db_path)
    if datasets is None or len(datasets) == 0:
        st.info("Belum ada dataset. Upload terlebih dahulu pada halaman Upload Dataset.")
        return

    choice = st.selectbox("Pilih dataset", datasets['id'].tolist())
    df = get_dataset_table(db_path, int(choice))
    if df.empty:
        st.error("Data tidak ditemukan di database.")
        return

    st.subheader("Preview 20 Baris")
    st.dataframe(df.head(20))

    st.subheader("Informasi Dataset")
    info = dataset_info(df)
    st.json(info)

    st.subheader("Missing Value per Kolom")
    st.write(pd.DataFrame.from_dict(info['missing'], orient='index', columns=['missing']))

    st.subheader("Tipe Data")
    st.write(pd.DataFrame.from_dict(info['dtypes'], orient='index', columns=['dtype']))

    st.markdown("---")
    st.subheader("Operasi Preprocessing")
    drop = st.checkbox("Hapus baris dengan missing value")
    encode = st.checkbox("Label Encoding otomatis (kolom kategorikal)")
    scale = st.checkbox("StandardScaler (normalisasi) untuk fitur numerik)")

    if st.button("Jalankan Preprocessing"):
        progress = st.progress(0)
        status = st.empty()
        df_proc = df.copy()
        progress.progress(15)
        status.write("Menyiapkan data...")
        if drop:
            status.write("Menghapus baris dengan missing value...")
            df_proc = drop_missing(df_proc)
            progress.progress(35)
        encoders = {}
        scaler = None
        if encode:
            status.write("Melakukan label encoding...")
            df_proc, encoders = label_encode(df_proc)
            progress.progress(60)
        if scale:
            status.write("Melakukan standard scaling...")
            df_proc, scaler = standard_scale(df_proc)
            progress.progress(80)

        st.success("Preprocessing selesai.")
        st.subheader("Preview Hasil Preprocessing")
        st.dataframe(df_proc.head(20))

        # Save processed table to sqlite as preprocessed_{dataset_id}
        conn = None
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            table_name = f"preprocessed_{choice}"
            df_proc.to_sql(table_name, conn, if_exists='replace', index=False)
            progress.progress(100)
            status.write("Hasil preprocessing tersimpan.")
            st.info(f"Hasil preprocessing disimpan ke tabel {table_name}.")
        except Exception as e:
            st.error(f"Gagal menyimpan hasil preprocessing: {e}")
        finally:
            if conn:
                conn.close()


if __name__ == '__main__':
    main()
