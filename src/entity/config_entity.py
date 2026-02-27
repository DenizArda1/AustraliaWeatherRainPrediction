import os
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

@dataclass
class DataValidationConfig:
    validation_status_file_path: str = os.path.join("artifacts", "validation_status.txt")
    schema_file_path: str = os.path.join("data_schema", "schema.yaml")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    train_arr_path: str = os.path.join("artifacts", "train_arr.npy")
    test_arr_path: str = os.path.join("artifacts", "test_arr.npy")

@dataclass
class ModelTrainingConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")