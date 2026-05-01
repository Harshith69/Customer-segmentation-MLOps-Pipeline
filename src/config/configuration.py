from pathlib import Path
from src.utils.common import read_yaml
from src.utils.logger import logger


class ConfigurationManager:
    def __init__(self, config_path: Path = Path("configs/config.yaml")):
        self.config = read_yaml(config_path)
        logger.info("Configuration loaded successfully")

    def get_config(self):
        return self.config