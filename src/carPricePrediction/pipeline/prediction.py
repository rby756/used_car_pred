import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from src.carPricePrediction.components.data_transformation import TargetEncodingTransformer



class PredictionPipeline:
    def __init__(self):
        self.target_encoder=TargetEncodingTransformer(cols=['short_carname'])
        self.target_encoder=joblib.load('artifacts/data_transformation/target_encoder.pkl')
        self.preprocessor = joblib.load('artifacts/data_transformation/preprocessor.joblib')
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
    
    def predict(self, data):
        data=self.target_encoder.transform(data)
        data = self.preprocessor.transform(data)
        prediction = self.model.predict(data)

        return prediction