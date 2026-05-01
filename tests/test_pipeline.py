import os
from src.pipelines.training_pipeline import run_pipeline


def test_pipeline_runs_successfully():
    try:
        run_pipeline()

        # Check artifacts
        assert os.path.exists("artifacts/raw/raw_data.csv")
        assert os.path.exists("artifacts/processed/processed_data.csv")
        assert os.path.exists("artifacts/models/kmeans_model.pkl")

    except Exception as e:
        assert False, f"Pipeline failed with error: {e}"