import numpy as np
import pandas as pd
import os
import sys

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import get_models_and_params,save_obj
from src.entity.config_entity import ModelTrainingConfig
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import RandomizedSearchCV


class ModelTrainer:
    def __init__(self,train_arr:np.ndarray,test_arr:np.ndarray):
        self.model_training_config = ModelTrainingConfig()
        self.X_train = train_arr[:,:-1]
        self.y_train = train_arr[:,-1]
        self.X_test = test_arr[:,:-1]
        self.y_test = test_arr[:,-1]

    def evaluate_model(self,model,model_name: str):
        y_test_preds = model.predict(self.X_test)
        y_train_preds = model.predict(self.X_train)
        acc_score_train = accuracy_score(self.y_train,y_train_preds)
        acc_score_test = accuracy_score(self.y_test,y_test_preds)

        report = classification_report(self.y_test,y_test_preds)
        logging.info(f"Model: {model_name} | Training Accuracy: {acc_score_train} | Testing Accuracy: {acc_score_test}")
        return acc_score_test

    def initiate_model_trainer(self):
        logging.info("Training Model")
        try:
            models,param_grid = get_models_and_params()
            model_report: dict = {}
            for model_name, model in models.items():
                params = param_grid[model_name]

                rs = RandomizedSearchCV(estimator=model,param_distributions=params,cv=3,verbose=1,n_jobs=1,n_iter=10,random_state=42)
                rs.fit(self.X_train,self.y_train)

                best_model = rs.best_estimator_
                test_model_score = self.evaluate_model(best_model,model_name)
                model_report[model_name] = {
                    "Model": best_model,
                    "Score": test_model_score
                }

            best_model_name = max(model_report, key=lambda k: model_report[k]["Score"])
            best_model_score = model_report[best_model_name]["Score"]
            best_model = model_report[best_model_name]["Model"]
            if best_model_score < 0.6:
                raise CustomException("All models are unsuccessful",sys)

            save_obj(self.model_training_config.model_path,best_model)
            return best_model_score
        except Exception as e:
            raise CustomException(e,sys)


