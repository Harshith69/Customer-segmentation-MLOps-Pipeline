import os
import pandas as pd
import pymongo
from dotenv import load_dotenv
from src.utils.logger import logger


class MongoLoader:
    def __init__(self, env_path: str = "credentials/mongodb.env"):
        load_dotenv(env_path)

        self.connection_url = os.getenv("CONNECTION_URL")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")

        if not all([self.connection_url, self.db_name, self.collection_name]):
            raise ValueError("MongoDB environment variables are missing")

        logger.info("MongoLoader initialized")

    def fetch_data(self) -> pd.DataFrame:
        try:
            client = pymongo.MongoClient(self.connection_url)
            db = client[self.db_name]
            collection = db[self.collection_name]

            data = list(collection.find())
            df = pd.DataFrame(data)

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            logger.info(f"Fetched {len(df)} records from MongoDB")

            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise