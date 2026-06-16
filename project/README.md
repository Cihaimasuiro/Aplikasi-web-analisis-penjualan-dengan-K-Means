# Aplikasi Segmentasi Pelanggan (K-Means)

Instruksi singkat:

- Install dependencies:

```
pip install -r requirements.txt
```

- Jalankan aplikasi:

```
streamlit run app.py
```

Folder utama:
- `app.py` - Halaman utama
- `pages/` - Halaman multipage Streamlit: dashboard, preprocessing, clustering, evaluation, visualization
- `utils/` - utilitas database, preprocessing, clustering, evaluation, visualization
- `database/database.db` - SQLite database (akan dibuat otomatis)
- `uploads/` - folder untuk menyimpan file upload (opsional)
- `exports/` - folder untuk hasil ekspor

Catatan:
- Implementasi ini modular dan siap dikembangkan.
- Untuk fitur export CSV, gunakan pandas `to_csv` dari hasil clustering.
