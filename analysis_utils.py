import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def load_data(path):
    """Load the student CSV (assumes semicolon separator if present)."""
    # Try common separators: comma first (the dataset uses commas), then semicolon
    try:
        df = pd.read_csv(path, sep=',')
        # if parsing resulted in a single column containing commas, try semicolon
        if df.shape[1] == 1 and df.columns[0].count(',') > 3:
            df = pd.read_csv(path, sep=';')
    except Exception:
        # fallback to pandas auto-detect
        df = pd.read_csv(path)
    return df


def clean_and_encode(df, target='G3', drop_cols=None):
    """Basic cleaning: handle missing values, encode categoricals.
    Returns X (DataFrame), y (Series), and encoder for later use.
    """
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # If target missing, drop
    df = df.dropna(subset=[target])

    # Separate target
    y = df[target]
    X = df.drop(columns=[target])

    # Simple imputation for numeric
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    # Impute numeric with median
    num_imputer = SimpleImputer(strategy='median')
    if len(numeric_cols) > 0:
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    # Fill categorical with 'missing' and one-hot encode
    X[cat_cols] = X[cat_cols].fillna('missing')
    # Use pandas get_dummies for simplicity
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def train_test_split_df(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
