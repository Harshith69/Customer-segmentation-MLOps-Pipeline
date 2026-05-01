import pandas as pd
from src.utils.logger import logger


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning")

    # Drop rows with critical nulls
    df = df.dropna(subset=["customer_id", "order_date", "revenue"])

    # Convert types
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Remove invalid values
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]

    logger.info(f"Data cleaned. Remaining rows: {len(df)}")

    return df