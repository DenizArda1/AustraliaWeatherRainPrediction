import os
import sys
import pandas as pd
import numpy as np
import yaml

from src.entity.config_entity import DataValidationConfig
from src.exception.exception import CustomException
from src.logger.logger import logging

class DataValidation:
    def __init__(self, train_path: str, test_path: str):
        self.validation_config = DataValidationConfig()
        self.train_path = train_path
        self.test_path = test_path

    def read_schema(self)->list:
        try:
            with open(self.validation_config.schema_file_path, "r") as file:
                schema = yaml.safe_load(file)
            columns_data = schema.get("columns", [])
            if isinstance(columns_data, list):
                expected_columns = [list(col.keys())[0] for col in columns_data]
            else:
                expected_columns = list(columns_data.keys())

            return expected_columns
        except Exception as e:
            raise CustomException(e,sys)

    def validate_columns(self):
        try:
            validation_status = True
            train_data = pd.read_csv(self.train_path)
            test_data = pd.read_csv(self.test_path)

            train_cols = list(train_data.columns)
            test_cols = list(test_data.columns)
            expected_cols = self.read_schema()

            # Train Data Validation
            for col in expected_cols:
                if col not in train_cols:
                    validation_status = False
                    logging.warning(f"{col} not in training data")

            # Test Data Validation
            for col in expected_cols:
                if col not in test_cols:
                    validation_status = False
                    logging.warning(f"{col} not in testing data")

            status_file_dir = os.path.dirname(self.validation_config.validation_status_file_path)
            os.makedirs(status_file_dir, exist_ok=True)

            with open(self.validation_config.validation_status_file_path,'w') as file:
                file.write(f"Validation Status: {validation_status}")
            if validation_status:
                logging.info("Validation Status: OK")
            else:
                logging.info("Validation Status: FAILED")
            return validation_status
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_validation(self)->bool:
        logging.info("Initiating Data Validation")
        try:
            status = self.validate_columns()
            if not status:
                error_message = f"Validation Failed!"
                logging.critical(error_message)
                raise Exception(error_message,sys)
            logging.info("Data Validation Completed")
            return status
        except Exception as e:
            raise CustomException(e,sys)