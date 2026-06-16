import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict


# Get dataset info: dtypes, missing counts
def dataset_info(df: pd.DataFrame) -> Dict:
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing": df.isnull().sum().to_dict(),
    }
    return info


# Drop missing values (row-wise) and return result
def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)


# Automatic label encoding for categorical/object columns
def label_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders = {}
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        try:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
        except Exception:
            continue
    return df_enc, encoders


# Standard scaling for numeric columns
def standard_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df_scaled = df.copy()
    num_cols = df_scaled.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    return df_scaled, scaler