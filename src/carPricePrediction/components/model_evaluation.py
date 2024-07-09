import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.carPricePrediction.entity.config_entity import ModelEvaluationConfig
from src.carPricePrediction import logger
from src.carPricePrediction.utils.common import read_yaml, create_directories, save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, y_test, y_pred):
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        return mae,mse,rmse,r2
    

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():

            predictions = model.predict(test_x)

            (mae,mse,rmse,r2) = self.eval_metrics(test_y, predictions)
            
            # Saving metrics as local
            scores = {"mae": mae, "mse": mse, "rmse": rmse, "r2_score": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("mae",mae)
            mlflow.log_metric("mse",mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestRegressor")
            else:
                mlflow.sklearn.log_model(model, "model")