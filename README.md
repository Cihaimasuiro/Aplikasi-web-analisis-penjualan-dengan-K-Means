# Dasbor K-Means Penjualan Ritel

Aplikasi web analisis data penjualan ritel berbasis pengelompokan K-Means dan analisis RFM.
Dibuat sebagai pendukung paper: **"Penerapan Algoritma K-Means pada Sistem Analisis Data Penjualan Ritel Berbasis Web"**.

## Fitur
- Unggah CSV data transaksi penjualan
- Deteksi otomatis kolom (ID Pelanggan, Tanggal, Jumlah Barang, Harga, Faktur, Produk)
- Dasbor ringkasan penjualan (pendapatan, tren bulanan, produk terlaris)
- Segmentasi pelanggan RFM dengan K-Means (implementasi murni JavaScript)
- Metode siku untuk membantu menentukan nilai k optimal
- Visualisasi diagram sebar, diagram pai, dan tabel karakteristik kelompok
- **Ringkasan eksekutif otomatis** (segmen kontributor pendapatan terbesar dan segmen berisiko kehilangan pelanggan)
- **Interpretasi detail per segmen**: deskripsi karakteristik RFM (tingkat Tinggi/Sedang/Rendah relatif terhadap rata-rata keseluruhan) beserta persentase populasi dan kontribusi pendapatan
- **Rekomendasi strategi bisnis dan pemasaran** otomatis per segmen berdasarkan kombinasi tingkat kebaruan transaksi, frekuensi transaksi, dan nilai belanja
- Unduh hasil segmentasi per pelanggan ke CSV
- Unduh ringkasan interpretasi dan rekomendasi per segmen ke CSV (untuk lampiran laporan/skripsi)

## Teknologi
- React 18 + Vite
- Recharts (visualisasi)
- PapaParse (membaca CSV)
- K-Means: implementasi JavaScript dengan inisialisasi K-Means++

## Cara Jalankan Lokal

```bash
npm install
npm run dev
```

## Publikasi ke Vercel

1. Unggah proyek ini ke GitHub
2. Buka https://vercel.com, lalu pilih impor repositori
3. Vercel otomatis mendeteksi Vite, lalu klik tombol publikasi
4. Selesai!

## Format CSV yang Didukung

Kolom (nama bebas, dipetakan saat unggah):
- **ID Pelanggan** (wajib): identitas unik pelanggan
- **Tanggal Transaksi** (wajib): format YYYY-MM-DD, DD/MM/YYYY, dll
- **Jumlah Barang** (wajib): angka positif
- **Harga Satuan** (wajib): angka positif
- **ID Transaksi/Faktur** (opsional): untuk frekuensi transaksi yang lebih akurat
- **Nama Produk** (opsional): untuk grafik produk terlaris
