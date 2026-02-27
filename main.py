from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer


path = "/Users/denizardasanal/AustraliaWeatherRainPrediction/data/data.csv"
data = DataIngestion(path)
train_data_path, test_data_path = data.initiate_data_ingestion()

validation = DataValidation(train_data_path, test_data_path)
validation.initiate_data_validation()

transform = DataTransformation(train_data_path, test_data_path)
train_arr, test_arr, preprocessor_path = transform.initiate_data_transformation()

model = ModelTrainer(train_arr, test_arr)
best_score = model.initiate_model_trainer()
print(best_score)
