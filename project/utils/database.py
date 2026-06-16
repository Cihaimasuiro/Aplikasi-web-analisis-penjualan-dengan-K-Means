import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any


# Initialize database and required tables
def init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_date TEXT,
            rows INTEGER,
            columns INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clustering_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            k_value INTEGER,
            silhouette_score REAL,
            davies_bouldin_index REAL,
            calinski_harabasz_score REAL,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS selected_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            features TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# Save uploaded dataframe to SQLite as table data_{dataset_id}
def save_dataset_to_db(df: pd.DataFrame, filename: str, db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        upload_date = datetime.utcnow().isoformat()
        rows, cols = df.shape
        cur.execute(
            "INSERT INTO datasets (filename, upload_date, rows, columns) VALUES (?, ?, ?, ?)",
            (filename, upload_date, rows, cols),
        )
        dataset_id = cur.lastrowid
        table_name = f"data_{dataset_id}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
        return dataset_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_datasets(db_path: str):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM datasets ORDER BY id DESC", conn)
    conn.close()
    return df


def get_dataset_table(db_path: str, dataset_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    table_name = f"data_{dataset_id}"
    try:
        df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def save_selected_features(db_path: str, dataset_id: int, features: List[str]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO selected_features (dataset_id, features, created_at) VALUES (?, ?, ?)",
        (dataset_id, ",".join(features), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_last_selected_features(db_path: str, dataset_id: int):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT features FROM selected_features WHERE dataset_id=? ORDER BY id DESC LIMIT 1",
        (dataset_id,),
    )
    r = cur.fetchone()
    conn.close()
    if r:
        return r[0].split(",")
    return []


def save_clustering_result(db_path: str, dataset_id: int, k: int, silhouette: float, dbi: float, ch: float) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO clustering_results (dataset_id, k_value, silhouette_score, davies_bouldin_index, calinski_harabasz_score, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, k, silhouette, dbi, ch, datetime.utcnow().isoformat()),
    )
    result_id = cur.lastrowid
    conn.commit()
    conn.close()
    return result_id


def get_last_clustering(db_path: str) -> Dict[str, Any] | None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM clustering_results ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    conn.close()
    if r:
        keys = ["id", "dataset_id", "k_value", "silhouette_score", "davies_bouldin_index", "calinski_harabasz_score", "created_at"]
        return dict(zip(keys, r))
    return None


def get_clustering_history(db_path: str, limit: int | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            cr.id,
            cr.dataset_id,
            d.filename,
            cr.k_value,
            cr.silhouette_score,
            cr.davies_bouldin_index,
            cr.calinski_harabasz_score,
            cr.created_at
        FROM clustering_results cr
        LEFT JOIN datasets d ON d.id = cr.dataset_id
        ORDER BY cr.id DESC
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_clustering_result(db_path: str, result_id: int) -> Dict[str, Any] | None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            cr.id,
            cr.dataset_id,
            d.filename,
            cr.k_value,
            cr.silhouette_score,
            cr.davies_bouldin_index,
            cr.calinski_harabasz_score,
            cr.created_at
        FROM clustering_results cr
        LEFT JOIN datasets d ON d.id = cr.dataset_id
        WHERE cr.id=?
        """,
        (result_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = [
        "id",
        "dataset_id",
        "filename",
        "k_value",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_score",
        "created_at",
    ]
    return dict(zip(keys, row))


def get_best_k_recommendations(db_path: str, dataset_id: int | None = None) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    where_clause = "WHERE dataset_id=?" if dataset_id is not None else ""
    params = (dataset_id,) if dataset_id is not None else ()

    best_silhouette = pd.read_sql_query(
        f"""
        SELECT * FROM clustering_results
        {where_clause}
        ORDER BY silhouette_score DESC
        LIMIT 1
        """,
        conn,
        params=params,
    )
    best_dbi = pd.read_sql_query(
        f"""
        SELECT * FROM clustering_results
        {where_clause}
        ORDER BY davies_bouldin_index ASC
        LIMIT 1
        """,
        conn,
        params=params,
    )
    conn.close()
    return {
        "silhouette": best_silhouette.iloc[0].to_dict() if not best_silhouette.empty else None,
        "davies_bouldin": best_dbi.iloc[0].to_dict() if not best_dbi.empty else None,
    }


def get_clustering_table(db_path: str, dataset_id: int, k: int, limit: int | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM clustering_{int(dataset_id)}_k{int(k)}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_clustering_k_values(db_path: str, dataset_id: int) -> List[int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT k_value FROM clustering_results WHERE dataset_id=? ORDER BY k_value",
        (dataset_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]
