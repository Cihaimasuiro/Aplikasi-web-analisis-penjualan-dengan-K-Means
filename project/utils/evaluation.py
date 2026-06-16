from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import pandas as pd
from typing import Dict, Any


# Compute evaluation metrics for clustering
def compute_scores(X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    scores = {}
    try:
        scores["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        scores["silhouette"] = float('nan')
    try:
        scores["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        scores["davies_bouldin"] = float('nan')
    try:
        scores["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        scores["calinski_harabasz"] = float('nan')
    return scores


# Provide textual interpretation for scores
def interpret_scores(scores: Dict[str, float]) -> Dict[str, str]:
    interp = {}
    s = scores.get("silhouette", float('nan'))
    if not np.isnan(s):
        if s > 0.6:
            interp["silhouette"] = "Kualitas cluster sangat baik karena mendekati 1."
        elif s > 0.4:
            interp["silhouette"] = "Kualitas cluster baik sampai sedang."
        else:
            interp["silhouette"] = "Kualitas cluster lemah; pertimbangkan jumlah cluster lain atau fitur berbeda."
    else:
        interp["silhouette"] = "Silhouette tidak dapat dihitung."

    db = scores.get("davies_bouldin", float('nan'))
    if not np.isnan(db):
        if db < 0.5:
            interp["davies_bouldin"] = "Davies-Bouldin rendah: cluster terpisah dengan baik."
        elif db < 1.0:
            interp["davies_bouldin"] = "Davies-Bouldin moderat."
        else:
            interp["davies_bouldin"] = "Davies-Bouldin tinggi: cluster mungkin tumpang tindih."
    else:
        interp["davies_bouldin"] = "Davies-Bouldin tidak dapat dihitung."

    ch = scores.get("calinski_harabasz", float('nan'))
    if not np.isnan(ch):
        interp["calinski_harabasz"] = "Semakin tinggi skor, semakin terpisah cluster dibandingkan variansi internal."
    else:
        interp["calinski_harabasz"] = "Calinski-Harabasz tidak dapat dihitung."

    return interp


def _pdf_text(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return text.encode("latin-1", "replace").decode("latin-1")


def build_evaluation_pdf(result: Dict[str, Any], interpretation: Dict[str, str]) -> bytes:
    lines = [
        "Laporan Evaluasi Clustering",
        "",
        f"ID Hasil: {result.get('id', '-')}",
        f"Dataset: {result.get('filename') or result.get('dataset_id', '-')}",
        f"K: {result.get('k_value', '-')}",
        f"Davies-Bouldin Index: {result.get('davies_bouldin_index', '-')}",
        f"Calinski-Harabasz Score: {result.get('calinski_harabasz_score', '-')}",
        f"Dibuat: {result.get('created_at', '-')}",
        "",
        "Interpretasi",
    ]
    for key, value in interpretation.items():
        lines.append(f"{key}: {value}")

    content_lines = ["BT", "/F1 12 Tf", "50 790 Td", "16 TL"]
    for index, line in enumerate(lines):
        escaped = _pdf_text(line)
        if index == 0:
            content_lines.append("/F1 16 Tf")
        elif index == 1:
            content_lines.append("/F1 12 Tf")
        content_lines.append(f"({escaped}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode("ascii")
    )
    return bytes(pdf)
