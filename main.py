from src.config.configuration import ConfigurationManager
from src.utils.logger import logger


def main():
    logger.info("Pipeline started")

    config = ConfigurationManager().get_config()
    logger.info(f"Loaded config: {config}")

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()

from src.pipelines.training_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()