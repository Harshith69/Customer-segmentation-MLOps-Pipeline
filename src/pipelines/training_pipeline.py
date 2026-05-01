from src.data_ingestion.ingestion import DataIngestion
from src.data_validation.validation import DataValidation
from src.data_transformation.data_transformation import DataTransformation
from src.model_training.train import ModelTrainer
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger


def run_pipeline():
    logger.info("Pipeline execution started")

    config = ConfigurationManager().get_config()

    # Step 1: Ingestion
    ingestion = DataIngestion(config)
    raw_data_path = ingestion.run()

    # Step 2: Validation
    validator = DataValidation(config)
    if not validator.run(raw_data_path):
        raise Exception("Validation failed")

    # Step 3: Transformation
    transformer = DataTransformation(config)
    processed_data_path, scaler = transformer.run(raw_data_path)

    logger.info(f"Processed data available at: {processed_data_path}")

    logger.info("Pipeline execution completed")

    # Step 4: Model Training
    trainer = ModelTrainer(config)
    model_path, metrics, run_id = trainer.train(processed_data_path)

    logger.info(f"Model stored at: {model_path}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"MLflow run_id: {run_id}")