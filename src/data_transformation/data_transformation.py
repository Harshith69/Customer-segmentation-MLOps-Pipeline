import os
from pathlib import Path
import pandas as pd

from src.data_transformation.preprocessing import clean_data
from src.data_transformation.feature_engineering import create_customer_features
from src.data_transformation.transformation import transform_features
from src.utils.logger import logger


class DataTransformation:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["artifacts_root"]) / "processed"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, data_path: Path):
        try:
            logger.info("Starting data transformation")

            df = pd.read_csv(data_path)

            df = clean_data(df)
            df_features = create_customer_features(df)
            df_transformed, scaler = transform_features(df_features)

            output_path = self.output_dir / "processed_data.csv"
            df_transformed.to_csv(output_path, index=False)

            logger.info(f"Processed data saved at {output_path}")

            return output_path, scaler

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise