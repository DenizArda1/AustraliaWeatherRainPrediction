import yaml
import os
import sys
import numpy as np
import pickle
import dill
from sklearn.ensemble import RandomForestClassifier

from src.exception.exception import CustomException
from src.logger.logger import logging
from xgboost import XGBClassifier

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)

def write_yaml_file(file_path:str, content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e,sys)

def save_obj(file_path:str, obj:object)->None:
    try:
        logging.info("saving obj")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_object:
            pickle.dump(obj, file_object)
        logging.info("obj saved")
    except Exception as e:
        raise CustomException(e,sys)

def load_obj(file_path:str)->object:
    try:
        logging.info("loading obj")
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist")
        with open(file_path, 'rb') as file_object:
            return pickle.load(file_object)
    except Exception as e:
        raise CustomException(e,sys)

def save_numpy_arr_data(file_path:str, arr:np.ndarray):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_object:
            np.save(file_object, arr)
    except Exception as e:
        raise CustomException(e,sys)

def load_numpy_arr_data(file_path:str)->np.ndarray:
    try:
        with open(file_path, 'rb') as file_object:
            return np.load(file_object)
    except Exception as e:
        raise CustomException(e,sys)

def get_models_and_params():
    models = {
        "RandomForestClassifier": RandomForestClassifier(n_jobs=1),
        "XGBClassifier": XGBClassifier(n_jobs=1)
    }
    param_grid = {
        "RandomForestClassifier": {
            "criterion": ["gini", "entropy"],
            "n_estimators": [16,32,64]
        },
        "XGBClassifier":{
            "learning_rate":[0.05,0.1,0.3],
            "max_depth":[3,5,7],
            "n_estimators":[100,200],
            "subsample":[0.8,1.0],
            "colsample_bytree":[0.8,1.0]
        }
    }
    return models,param_grid


