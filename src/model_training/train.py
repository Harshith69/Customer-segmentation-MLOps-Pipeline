import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

from src.model_training.evaluate import ModelEvaluator
from src.model_tracking.mlflow_tracker import MLflowTracker
from src.utils.logger import logger


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_dir = Path(config["artifacts_root"]) / "models"
        os.makedirs(self.model_dir, exist_ok=True)

        self.evaluator = ModelEvaluator()
        self.tracker = MLflowTracker(config)

    def train(self, data_path: Path):
        try:
            logger.info("Starting model training")

            df = pd.read_csv(data_path)
            X = df.drop(columns=["customer_id"])

            k = self.config["model_training"]["n_clusters"]
            random_state = self.config["model_training"]["random_state"]

            model = KMeans(n_clusters=k, random_state=random_state)
            model.fit(X)

            # Evaluation
            metrics = self.evaluator.evaluate(model, X)

            # Tracking
            params = {
                "n_clusters": k,
                "random_state": random_state
            }

            run_id = self.tracker.log(
                model=model,
                params=params,
                metrics=metrics,
                X_shape=X.shape
            )

            # Save locally
            model_path = self.model_dir / "kmeans_model.pkl"
            joblib.dump(model, model_path)

            logger.info(f"Model saved at {model_path}")

            return model_path, metrics, run_id

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise