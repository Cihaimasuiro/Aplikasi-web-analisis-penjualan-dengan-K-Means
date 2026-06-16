# Retail K-Means Dashboard

Aplikasi web analisis data penjualan retail berbasis K-Means Clustering (RFM Analysis).
Dibuat sebagai pendukung paper: **"Penerapan Algoritma K-Means pada Sistem Analisis Data Penjualan Retail Berbasis Web"**.

## Fitur
- Upload CSV data transaksi penjualan
- Auto-detect kolom (CustomerID, Tanggal, Qty, Harga, Invoice, Produk)
- Dashboard ringkasan penjualan (revenue, tren bulanan, produk terlaris)
- Segmentasi pelanggan RFM dengan K-Means (implementasi murni JavaScript)
- Elbow Method untuk menentukan nilai k optimal
- Visualisasi scatter plot, pie chart, dan tabel karakteristik cluster
- Download hasil segmentasi ke CSV

## Tech Stack
- React 18 + Vite
- Recharts (visualisasi)
- PapaParse (parsing CSV)
- K-Means: implementasi custom JS dengan K-Means++ initialization

## Cara Jalankan Lokal

```bash
npm install
npm run dev
```

## Deploy ke Vercel

1. Push project ini ke GitHub
2. Buka https://vercel.com → Import repository
3. Vercel otomatis mendeteksi Vite → klik Deploy
4. Selesai!

## Format CSV yang Didukung

Kolom (nama bebas, dipetakan saat upload):
- **ID Pelanggan** (wajib): identitas unik pelanggan
- **Tanggal Transaksi** (wajib): format YYYY-MM-DD, DD/MM/YYYY, dll
- **Jumlah/Qty** (wajib): angka positif
- **Harga Satuan** (wajib): angka positif
- **ID Invoice** (opsional): untuk Frequency yang lebih akurat
- **Nama Produk** (opsional): untuk grafik produk terlaris
