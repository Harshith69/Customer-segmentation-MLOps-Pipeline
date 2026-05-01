from dotenv import load_dotenv
import os

# Load env
load_dotenv(r"C:\Users\Harshith N\Documents\Personal_projects\Customer-segmentation-MLOps-Pipeline\credentials\dagshub.env")

# Force set (important)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

import mlflow
import mlflow.sklearn
from src.utils.logger import logger


class MLflowTracker:
    def __init__(self, config: dict):
        self.config = config

        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    def log(self, model, params: dict, metrics: dict, X_shape, run_name=None):
        try:
            with mlflow.start_run(run_name=run_name):

                # Params
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # Dataset info
                mlflow.log_param("num_rows", X_shape[0])
                mlflow.log_param("num_features", X_shape[1])

                # Metrics
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                # Log + Register together
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.config["mlflow"]["registered_model_name"]
                )

                logger.info("Model logged and registered successfully")

        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            raise