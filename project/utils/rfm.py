"""
RFM (Recency, Frequency, Monetary) computation utilities.
Detects the right columns automatically or lets caller specify them.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any


CUSTOMER_HINTS   = ['customerid','customer_id','pelanggan','id_pelanggan','customer']
DATE_HINTS       = ['invoicedate','orderdate','order_date','date','tanggal','tgl','transaction_date']
INVOICE_HINTS    = ['invoiceno','invoice_no','orderid','order_id','no_transaksi','invoice']
MONETARY_HINTS   = ['totalspent','total_spent','total','amount','harga','revenue','monetary']
QUANTITY_HINTS   = ['quantity','qty','jumlah']
PRICE_HINTS      = ['unitprice','unit_price','price','harga_satuan','hargasatuan']


def _norm(s: str) -> str:
    return s.lower().replace(' ', '').replace('_', '').replace('-', '')


def _find(headers: list[str], hints_norm: list[str]) -> Optional[str]:
    for h in headers:
        norm_h = _norm(h)
        for hint_norm in hints_norm:
            if norm_h == hint_norm:
                return h
    return None


def detect_rfm_columns(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    # Pre-normalize hints
    norm_customer_hints = [_norm(h) for h in CUSTOMER_HINTS]
    norm_date_hints = [_norm(h) for h in DATE_HINTS]
    norm_invoice_hints = [_norm(h) for h in INVOICE_HINTS]
    norm_monetary_hints = [_norm(h) for h in MONETARY_HINTS]
    norm_quantity_hints = [_norm(h) for h in QUANTITY_HINTS]
    norm_price_hints = [_norm(h) for h in PRICE_HINTS]

    return {
        'customer': _find(cols, norm_customer_hints),
        'date':     _find(cols, norm_date_hints),
        'invoice':  _find(cols, norm_invoice_hints),
        'monetary': _find(cols, norm_monetary_hints),
        'quantity': _find(cols, norm_quantity_hints),
        'price':    _find(cols, norm_price_hints),
    }


def compute_rfm(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    monetary_col: Optional[str] = None,
    quantity_col: Optional[str] = None,
    price_col: Optional[str] = None,
    invoice_col: Optional[str] = None,
    reference_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Returns (rfm_df, diagnostics).

    rfm_df  : DataFrame dengan kolom CustomerID, Recency, Frequency, Monetary.
    diagnostics : dict berisi info setiap tahap untuk debugging UI.

    Monetary berasal dari:
      1. monetary_col langsung (misal TotalSpent)
      2. quantity_col * price_col
      3. fallback: kolom numerik pertama
    Frequency berasal dari:
      1. invoice_col (unique invoices per customer)
      2. jumlah baris per customer
    """
    diag = {}
    df = df.copy()

    # ── Step 1: validasi kolom customer ──────────────────────────────────────
    if customer_col not in df.columns:
        raise ValueError(f"Kolom customer '{customer_col}' tidak ditemukan.")
    diag['step1_total_rows']        = len(df)
    diag['step1_unique_customers']  = df[customer_col].nunique()
    diag['step1_customer_col']      = customer_col
    diag['step1_sample_customers']  = df[customer_col].dropna().unique()[:5].tolist()

    # Drop baris tanpa customer ID
    before = len(df)
    df = df.dropna(subset=[customer_col])
    diag['step1_rows_dropped_no_customer'] = before - len(df)

    # ── Step 2: parse tanggal ─────────────────────────────────────────────────
    diag['step2_date_col']          = date_col
    diag['step2_sample_raw_dates']  = df[date_col].dropna().astype(str).unique()[:5].tolist()

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    n_before_date = len(df)
    n_failed_date = df[date_col].isna().sum()
    df = df.dropna(subset=[date_col])
    diag['step2_rows_failed_date_parse'] = int(n_failed_date)
    diag['step2_rows_after_date_drop']   = len(df)
    diag['step2_date_range']             = (
        f"{df[date_col].min().date()} → {df[date_col].max().date()}"
        if len(df) > 0 else "—"
    )

    if len(df) == 0:
        raise ValueError(
            "Semua baris dibuang karena kolom tanggal gagal di-parse. "
            "Pastikan format tanggal konsisten (contoh: YYYY-MM-DD atau DD/MM/YYYY)."
        )

    if reference_date is None:
        reference_date = df[date_col].max() + pd.Timedelta(days=1)
    assert reference_date is not None
    diag['step2_reference_date'] = str(reference_date.date())

    # ── Step 3: hitung nilai monetary per baris ───────────────────────────────
    if monetary_col and monetary_col in df.columns:
        df['_amount'] = pd.to_numeric(df[monetary_col], errors='coerce')
        diag['step3_monetary_source'] = f"kolom langsung: '{monetary_col}'"
    elif quantity_col and price_col and quantity_col in df.columns and price_col in df.columns:
        df['_amount'] = (
            pd.to_numeric(df[quantity_col], errors='coerce').fillna(0) *
            pd.to_numeric(df[price_col], errors='coerce').fillna(0)
        )
        diag['step3_monetary_source'] = f"qty × harga ({quantity_col} × {price_col})"
    else:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        fallback = num_cols[0] if num_cols else None
        df['_amount'] = pd.to_numeric(df[fallback], errors='coerce') if fallback else 1
        diag['step3_monetary_source'] = f"fallback: '{fallback}'" if fallback else "konstanta 1"

    n_nan_amount = df['_amount'].isna().sum()
    df['_amount'] = df['_amount'].fillna(0).clip(lower=0)
    diag['step3_rows_nan_amount'] = int(n_nan_amount)
    diag['step3_amount_stats'] = {
        'min': round(float(df['_amount'].min()), 2),
        'max': round(float(df['_amount'].max()), 2),
        'mean': round(float(df['_amount'].mean()), 2),
        'zero_pct': round(float((df['_amount'] == 0).mean() * 100), 1),
    }

    # ── Step 4: groupby dan hitung RFM ───────────────────────────────────────
    agg: dict = {
        'Recency':  (date_col, lambda x: (reference_date - x.max()).days),
        'Monetary': ('_amount', 'sum'),
    }
    if invoice_col and invoice_col in df.columns:
        agg['Frequency'] = (invoice_col, 'nunique')
        diag['step4_frequency_source'] = f"unique invoices dari '{invoice_col}'"
    else:
        agg['Frequency'] = (date_col, 'count')
        diag['step4_frequency_source'] = 'jumlah baris (count) per customer'

    rfm = df.groupby(customer_col).agg(**agg).reset_index()
    rfm = rfm.rename(columns={customer_col: 'CustomerID'})
    rfm['Recency']   = rfm['Recency'].astype(int)
    rfm['Frequency'] = rfm['Frequency'].astype(int)
    rfm['Monetary']  = rfm['Monetary'].round(2)

    diag['step4_customers_before_filter'] = len(rfm)

    # Filter: hanya buang Frequency = 0 (anomali data)
    rfm = rfm[rfm['Frequency'] > 0]
    diag['step4_customers_after_filter'] = len(rfm)

    diag['step4_rfm_stats'] = {
        'recency':   {'min': int(rfm['Recency'].min()),   'max': int(rfm['Recency'].max()),   'mean': round(float(rfm['Recency'].mean()), 1)},
        'frequency': {'min': int(rfm['Frequency'].min()), 'max': int(rfm['Frequency'].max()), 'mean': round(float(rfm['Frequency'].mean()), 1)},
        'monetary':  {'min': round(float(rfm['Monetary'].min()), 2), 'max': round(float(rfm['Monetary'].max()), 2), 'mean': round(float(rfm['Monetary'].mean()), 2)},
    }

    return rfm.reset_index(drop=True), diag


def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise R, F, M to [0, 1]. Returns scaled copy."""
    from sklearn.preprocessing import MinMaxScaler
    scaled = rfm.copy()
    cols = ['Recency', 'Frequency', 'Monetary']
    sc = MinMaxScaler()
    scaled[cols] = sc.fit_transform(scaled[cols])
    return scaled


def rfm_summary_table(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Build a rich cluster-level summary of RFM values.

    Returns DataFrame with columns:
      Cluster | Jumlah Pelanggan | % Pelanggan | Avg Recency (hari) |
      Avg Frequency | Avg Monetary (Rp) | Total Monetary (Rp) |
      Centroid R | Centroid F | Centroid M | Ranking Nilai Pelanggan
    """
    df = rfm.copy()
    df['Cluster'] = labels
    total = len(df)

    rows = []
    for c in sorted(df['Cluster'].unique()):
        cdf = df[df['Cluster'] == c]
        rows.append({
            'Cluster': int(c),
            'Jumlah Pelanggan': len(cdf),
            '% Pelanggan': round(len(cdf) / total * 100, 1),
            'Avg Recency (hari)': round(cdf['Recency'].mean(), 1),
            'Avg Frequency': round(cdf['Frequency'].mean(), 1),
            'Avg Monetary (Rp)': round(cdf['Monetary'].mean(), 2),
            'Total Monetary (Rp)': round(cdf['Monetary'].sum(), 2),
        })

    summary = pd.DataFrame(rows)

    # Ranking berdasarkan Customer Value:
    # tinggi F + tinggi M - tinggi R  →  semakin baik
    max_r = summary['Avg Recency (hari)'].max() or 1
    max_f = summary['Avg Frequency'].max() or 1
    max_m = summary['Avg Monetary (Rp)'].max() or 1
    summary['Skor Nilai Pelanggan'] = (
        summary['Avg Frequency'] / max_f +
        summary['Avg Monetary (Rp)'] / max_m -
        summary['Avg Recency (hari)'] / max_r
    ).round(4)
    summary['Ranking Nilai Pelanggan'] = summary['Skor Nilai Pelanggan'].rank(ascending=False).astype(int)

    return summary.sort_values('Cluster').reset_index(drop=True)
