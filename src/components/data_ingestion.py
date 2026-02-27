import pandas as pd
import numpy as np
import os
import sys

from src.exception.exception import CustomException
from src.logger.logger import logging

from src.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8

class DataIngestion:
    def __init__(self,input_path):
        self.data_ingestion_config = DataIngestionConfig()
        self.input_path = input_path
        self.data = None

    def load_data(self)->pd.DataFrame:
        try:
            self.data = pd.read_csv(self.input_path)
            logging.info("Data loaded successfully")
            return self.data
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            data = self.load_data()
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            train_data, test_data = train_test_split(data,train_size=TRAIN_SIZE,random_state=42)
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion initiated successfully")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)