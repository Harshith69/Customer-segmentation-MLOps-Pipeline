import os
from pathlib import Path
from src.data_ingestion.mongo_loader import MongoLoader
from src.utils.logger import logger


class DataIngestion:
    def __init__(self, config: dict):
        self.config = config
        self.artifacts_dir = Path(config["artifacts_root"]) / "raw"
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def run(self) -> Path:
        try:
            logger.info("Starting data ingestion")

            loader = MongoLoader()
            df = loader.fetch_data()

            output_path = self.artifacts_dir / "raw_data.csv"
            df.to_csv(output_path, index=False)

            logger.info(f"Data saved at {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise