import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logger
import joblib


def transform_features(df: pd.DataFrame):
    logger.info("Starting feature transformation")

    feature_cols = df.columns.drop("customer_id")

    # Log transform (handle skewness)
    df[feature_cols] = df[feature_cols].apply(lambda x: np.log1p(x))

    # Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    scaled_df["customer_id"] = df["customer_id"].values

    logger.info("Feature transformation completed")
    joblib.dump(scaler, "artifacts/models/scaler.pkl")
    return scaled_df, scaler