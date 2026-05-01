import pandas as pd
from datetime import datetime
from src.utils.logger import logger


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering")

    snapshot_date = df["order_date"].max()

    # RFM Features
    rfm = df.groupby("customer_id").agg({
        "order_date": lambda x: (snapshot_date - x.max()).days,  # Recency
        "order_id": "count",                                    # Frequency
        "revenue": "sum"                                        # Monetary
    }).rename(columns={
        "order_date": "recency",
        "order_id": "frequency",
        "revenue": "monetary"
    })

    # Additional Features
    extra = df.groupby("customer_id").agg({
        "quantity": "sum",
        "discount": "mean",
        "delivery_days": "mean",
        "customer_rating": "mean"
    })

    features = rfm.join(extra)

    logger.info("Feature engineering completed")

    return features.reset_index()