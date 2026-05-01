import yaml
from pathlib import Path


def read_yaml(file_path: Path) -> dict:
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def create_directories(paths: list):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)