from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from src.utils.logger import logger


class ModelEvaluator:
    def evaluate(self, model, X):
        try:
            logger.info("Starting model evaluation")

            clusters = model.predict(X)

            silhouette = silhouette_score(X, clusters)
            davies_bouldin = davies_bouldin_score(X, clusters)
            calinski_harabasz = calinski_harabasz_score(X, clusters)

            logger.info(f"Silhouette Score: {silhouette}")
            logger.info(f"Davies-Bouldin Score: {davies_bouldin}")
            logger.info(f"Calinski-Harabasz Score: {calinski_harabasz}")

            return {
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "calinski_harabasz_score": calinski_harabasz
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise