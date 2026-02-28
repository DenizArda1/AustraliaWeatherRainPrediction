import os
import sys
import pandas as pd
import numpy as np

from src.constants import TARGET_COLUMN
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import save_obj,save_numpy_arr_data
from src.entity.config_entity import DataTransformationConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class DataTransformation:
    def __init__(self,train_data_path,test_data_path):
        self.data_transformation_config = DataTransformationConfig()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def transformer_obj(self,num_cols: list, cat_cols: list):
        try:
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self):
        logging.info("Data Transformation Started")
        try:
            train_data = pd.read_csv(self.train_data_path)
            test_data = pd.read_csv(self.test_data_path)

            train_data['Date'] = pd.to_datetime(train_data['Date'])
            train_data['Year'] = train_data['Date'].dt.year
            train_data['Month'] = train_data['Date'].dt.month
            train_data['Day'] = train_data['Date'].dt.day
            train_data.drop(columns=['Date'], inplace=True)

            test_data['Date'] = pd.to_datetime(test_data['Date'])
            test_data['Year'] = test_data['Date'].dt.year
            test_data['Month'] = test_data['Date'].dt.month
            test_data['Day'] = test_data['Date'].dt.day
            test_data.drop(columns=['Date'], inplace=True)

            input_feature_train_data = train_data.drop(columns=[TARGET_COLUMN])
            input_feature_test_data = test_data.drop(columns=[TARGET_COLUMN])

            target_feature_train_data = train_data[TARGET_COLUMN]
            target_feature_test_data = test_data[TARGET_COLUMN]

            num_cols = input_feature_train_data.select_dtypes(exclude=["object","string"]).columns.tolist()
            cat_cols = input_feature_train_data.select_dtypes(include=["object","string"]).columns.tolist()

            preprocessor = self.transformer_obj(num_cols, cat_cols)

            transformed_input_feature_train_data = preprocessor.fit_transform(input_feature_train_data)
            transformed_input_feature_test_data = preprocessor.transform(input_feature_test_data)

            target_encoder = LabelEncoder()
            target_train_data_encoded = target_encoder.fit_transform(target_feature_train_data)
            target_test_data_encoded = target_encoder.transform(target_feature_test_data)

            train_arr = np.c_[transformed_input_feature_train_data, target_train_data_encoded]
            test_arr = np.c_[transformed_input_feature_test_data, target_test_data_encoded]

            save_obj(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            save_numpy_arr_data(self.data_transformation_config.train_arr_path, train_arr)
            save_numpy_arr_data(self.data_transformation_config.test_arr_path, test_arr)
            logging.info("Data Transformation Completed")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except CustomException as e:
            raise CustomException(e,sys)

