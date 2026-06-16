import streamlit as st
import pandas as pd
import os
from io import BytesIO
from utils.database import init_db, save_dataset_to_db
from utils.ui import setup_page


def main():
    st.set_page_config(page_title="Upload Dataset", layout="wide")
    setup_page()
    st.title("Upload Dataset (CSV)")

    db_path = os.path.join("database", "database.db")
    init_db(db_path)

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = os.path.basename(uploaded_file.name)
        try:
            df = pd.read_csv(BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            return

        st.subheader("Preview 20 baris")
        st.dataframe(df.head(20))

        st.write("Jumlah baris:", df.shape[0])
        st.write("Jumlah kolom:", df.shape[1])

        if st.button("Simpan ke Database"):
            try:
                # save csv to uploads folder
                os.makedirs('uploads', exist_ok=True)
                save_path = os.path.join('uploads', filename)
                with open(save_path, 'wb') as f:
                    f.write(file_bytes)

                dataset_id = save_dataset_to_db(df, filename, db_path)
                st.success(f"Dataset tersimpan dengan id {dataset_id}.")
            except Exception as e:
                st.error(f"Gagal menyimpan dataset: {e}")


if __name__ == '__main__':
    main()
