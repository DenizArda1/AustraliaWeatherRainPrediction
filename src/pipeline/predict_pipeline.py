import sys
import os
import pandas as pd

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_obj

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Prediction Pipeline is starting...")

            model = load_obj(file_path=self.model_path)
            preprocessor = load_obj(file_path=self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Date: str, Location: str, MinTemp: float, MaxTemp: float, Rainfall: float,
                 Evaporation: float, Sunshine: float, WindGustDir: str, WindGustSpeed: float,
                 WindDir9am: str, WindDir3pm: str, WindSpeed9am: float, WindSpeed3pm: float,
                 Humidity9am: float, Humidity3pm: float, Pressure9am: float, Pressure3pm: float,
                 Cloud9am: float, Cloud3pm: float, Temp9am: float, Temp3pm: float, RainToday: str):

        self.Date = Date
        self.Location = Location
        self.MinTemp = MinTemp
        self.MaxTemp = MaxTemp
        self.Rainfall = Rainfall
        self.Evaporation = Evaporation
        self.Sunshine = Sunshine
        self.WindGustDir = WindGustDir
        self.WindGustSpeed = WindGustSpeed
        self.WindDir9am = WindDir9am
        self.WindDir3pm = WindDir3pm
        self.WindSpeed9am = WindSpeed9am
        self.WindSpeed3pm = WindSpeed3pm
        self.Humidity9am = Humidity9am
        self.Humidity3pm = Humidity3pm
        self.Pressure9am = Pressure9am
        self.Pressure3pm = Pressure3pm
        self.Cloud9am = Cloud9am
        self.Cloud3pm = Cloud3pm
        self.Temp9am = Temp9am
        self.Temp3pm = Temp3pm
        self.RainToday = RainToday

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Date": [self.Date], "Location": [self.Location], "MinTemp": [self.MinTemp],
                "MaxTemp": [self.MaxTemp], "Rainfall": [self.Rainfall], "Evaporation": [self.Evaporation],
                "Sunshine": [self.Sunshine], "WindGustDir": [self.WindGustDir], "WindGustSpeed": [self.WindGustSpeed],
                "WindDir9am": [self.WindDir9am], "WindDir3pm": [self.WindDir3pm], "WindSpeed9am": [self.WindSpeed9am],
                "WindSpeed3pm": [self.WindSpeed3pm], "Humidity9am": [self.Humidity9am],
                "Humidity3pm": [self.Humidity3pm],
                "Pressure9am": [self.Pressure9am], "Pressure3pm": [self.Pressure3pm], "Cloud9am": [self.Cloud9am],
                "Cloud3pm": [self.Cloud3pm], "Temp9am": [self.Temp9am], "Temp3pm": [self.Temp3pm],
                "RainToday": [self.RainToday]
            }

            df = pd.DataFrame(custom_data_input_dict)

            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df.drop(columns=['Date'], inplace=True)

            logging.info("User info has successfully transformed into dataframe.")
            return df
        except Exception as e:
            raise CustomException(e, sys)