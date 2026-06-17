# web-data-processing-kategori-clustering-k-means

Web olah data untuk clustering menggunakan metode k-means.

Sistem ini memungkinkan perhitungan k-means secara langsung berbasis Python.
Dari sistem ini terdapat Input CSV secara manual, kemudian menentukan variabel yang ingin diolah.

**Fitur:**
- Penghapusan data yang kosong
- Pergantian data non-numerik ke numerik
- Perhitungan nilai K (jumlah cluster)
- Evaluasi hasil clustering
- Visualisasi grafik PCA (Principal Component Analysis)

**Cara Menjalankan Aplikasi:**

1.  **Clone Repository:**
    `ash
    git clone https://github.com/your-username/web-data-processing-kategori-clustering-k-means.git
    cd web-data-processing-kategori-clustering-k-means
    `

2.  **Instalasi Dependencies:**
    Pastikan Anda memiliki pip terinstal. Instal semua dependensi yang diperlukan dari 
equirements.txt:
    `ash
    pip install -r requirements.txt
    `

3.  **Menjalankan Aplikasi Streamlit:**
    Setelah semua dependensi terinstal, jalankan aplikasi menggunakan Streamlit:
    `ash
    streamlit run app.py
    `
    Aplikasi akan terbuka di browser default Anda (biasanya http://localhost:8501).

**Penggunaan Aplikasi:**

1.  **Upload CSV:** Unggah file CSV Anda melalui antarmuka aplikasi.
2.  **Pilih Variabel:** Pilih kolom/variabel yang ingin Anda gunakan untuk clustering.
3.  **Pre-processing Data:** Aplikasi akan secara otomatis menangani data kosong dan mengubah data non-numerik.
4.  **Tentukan Nilai K:** Masukkan nilai K yang diinginkan untuk algoritma k-means.
5.  **Lihat Hasil:** Lihat hasil clustering, evaluasi, dan visualisasi PCA.

**Kontribusi:**

Kontribusi disambut baik! Silakan ajukan pull request atau buka issue untuk saran dan perbaikan.