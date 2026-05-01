import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.common import read_yaml


class DataValidation:
    def __init__(self, config: dict):
        self.config = config
        self.schema = read_yaml(Path("configs/schema.yaml"))

    def validate_columns(self, df: pd.DataFrame) -> bool:
        expected = set(self.schema["required_columns"])
        actual = set(df.columns)

        missing = expected - actual

        if missing:
            logger.error(f"Missing columns: {missing}")
            return False

        logger.info("Column validation passed")
        return True

    def validate_dtypes(self, df: pd.DataFrame) -> bool:
        # Relaxed dtype validation (real-world friendly)
        for col in self.schema["numerical_columns"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"{col} should be numeric")
                return False

        logger.info("Dtype validation passed")
        return True

    def validate_nulls(self, df: pd.DataFrame) -> bool:
        null_counts = df.isnull().sum()

        if null_counts.any():
            logger.warning(f"Null values detected:\n{null_counts}")

        return True  # don't fail pipeline

    def validate_datetime(self, df: pd.DataFrame) -> bool:
        try:
            df["order_date"] = pd.to_datetime(df["order_date"])
            logger.info("order_date converted successfully")
            return True
        except Exception:
            logger.error("Invalid order_date format")
            return False

    def validate_business_rules(self, df: pd.DataFrame) -> bool:
        status = True

        # Quantity should be > 0
        if (df["quantity"] <= 0).any():
            logger.warning("Non-positive quantity detected")

        # Unit price must be positive
        if (df["unit_price"] <= 0).any():
            logger.error("Invalid unit_price (<=0)")
            status = False

        # Discount should be between 0 and 1
        if "discount" in df.columns:
            if ((df["discount"] < 0) | (df["discount"] > 1)).any():
                logger.error("Discount out of range [0,1]")
                status = False

        # Revenue consistency check
        expected_revenue = df["quantity"] * df["unit_price"] * (1 - df["discount"])
        mismatch = (df["revenue"] - expected_revenue).abs().mean()

        if mismatch > 1e-3:
            logger.warning("Revenue mismatch detected (possible data issue)")

        # Delivery days should be >= 0
        if (df["delivery_days"] < 0).any():
            logger.error("Negative delivery_days found")
            status = False

        return status

    def validate_duplicates(self, df: pd.DataFrame) -> bool:
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            logger.warning(f"{duplicates} duplicate rows found")

        return True

    def run(self, data_path: Path) -> bool:
        try:
            logger.info("Starting data validation")

            df = pd.read_csv(data_path)

            checks = [
                self.validate_columns(df),
                self.validate_dtypes(df),
                self.validate_nulls(df),
                self.validate_datetime(df),
                self.validate_business_rules(df),
                self.validate_duplicates(df),
            ]

            if all(checks):
                logger.info("Data validation passed")
                return True
            else:
                logger.error("Data validation failed")
                return False

        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise