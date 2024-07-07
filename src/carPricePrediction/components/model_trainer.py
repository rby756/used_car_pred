import pandas as pd
from src.carPricePrediction.entity.config_entity import ModelTrainerConfig
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    
    def train(self):

        print("train data path ",self.config)
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        rf_rgrs = RandomForestRegressor(n_estimators=self.config.n_estimators)
        rf_rgrs.fit(train_x, train_y)

        joblib.dump(rf_rgrs, os.path.join(self.config.root_dir, self.config.model_name))