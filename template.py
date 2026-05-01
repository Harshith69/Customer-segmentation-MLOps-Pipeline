import os
from pathlib import Path

# Base path = current directory
base_path = Path.cwd()

list_of_files = [

    # Root files
    "main.py",
    "requirements.txt",
    "setup.py",
    "README.md",

    # Configs
    "configs/config.yaml",
    "configs/schema.yaml",

    # Artifacts
    "artifacts/raw/.gitkeep",
    "artifacts/processed/.gitkeep",
    "artifacts/models/.gitkeep",
    "artifacts/reports/.gitkeep",

    # Data ingestion
    "src/data_ingestion/__init__.py",
    "src/data_ingestion/mongo_loader.py",
    "src/data_ingestion/ingestion.py",

    # Data validation
    "src/data_validation/__init__.py",
    "src/data_validation/validation.py",

    # Data transformation
    "src/data_transformation/__init__.py",
    "src/data_transformation/preprocessing.py",
    "src/data_transformation/feature_engineering.py",
    "src/data_transformation/transformation.py",

    # Model training
    "src/model_training/__init__.py",
    "src/model_training/train.py",
    "src/model_training/evaluate.py",

    # Model tracking
    "src/model_tracking/__init__.py",
    "src/model_tracking/mlflow_tracker.py",

    # Pipelines
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",

    # Utils
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/common.py",

    # Constants
    "src/constants/__init__.py",
    "src/constants/constants.py",

    # Entity
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",

    # Config handling
    "src/config/configuration.py",

    # Notebooks
    "notebooks/experiments.ipynb",

    # Tests
    "tests/test_pipeline.py",

    # Credentials
    "credentials/mongodb.env",
]


def create_project_structure():
    for file in list_of_files:
        filepath = base_path / file
        filedir = filepath.parent

        # Create directories
        os.makedirs(filedir, exist_ok=True)

        # Create file only if not exists or empty
        if not filepath.exists() or filepath.stat().st_size == 0:
            with open(filepath, "w") as f:
                pass
            print(f"Created: {filepath}")
        else:
            print(f"Skipped: {filepath}")


if __name__ == "__main__":
    create_project_structure()