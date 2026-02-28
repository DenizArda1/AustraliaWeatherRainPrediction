import os
import sys

from src.components.model_trainer import ModelTrainer
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.constants import DATA_PATH

class TrainingPipeline:
    def __init__(self):
        self.data_path = DATA_PATH
        self.train_data_path = None
        self.test_data_path = None
        self.train_arr = None
        self.test_arr = None

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion(self.data_path)
            self.train_data_path, self.test_data_path = data_ingestion.initiate_data_ingestion()
            return self.train_data_path, self.test_data_path
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_validation(self):
        try:
            data_validation = DataValidation(train_path=self.train_data_path, test_path=self.test_data_path)
            status = data_validation.initiate_data_validation()
            return status
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_transformation(self):
        try:
            data_transformation = DataTransformation(self.train_data_path, self.test_data_path)
            self.train_arr, self.test_arr,_ = data_transformation.initiate_data_transformation()
            return self.train_arr, self.test_arr
        except Exception as e:
            raise CustomException(e,sys)

    def start_model_training(self):
        try:
            model_trainer = ModelTrainer(self.train_arr, self.test_arr)
            score = model_trainer.initiate_model_trainer()
            return score
        except Exception as e:
            raise CustomException(e,sys)

    def run_pipeline(self):
        try:
            self.start_data_ingestion()
            self.start_data_validation()
            self.start_data_transformation()
            self.start_model_training()
        except Exception as e:
            logging.critical("Error during the training pipeline!")
            raise CustomException(e,sys)
