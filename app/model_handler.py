import joblib
import numpy as np
import pandas as pd

MODEL_PATHS = {
    "v1": "./models/model_v1.pkl",
    "v2": "./models/model_v2.pkl",
}

class ModelHandler:
    def __init__(self):
        self.models = {}
        self.columns = None
        for version, path in MODEL_PATHS.items():
            bundle = joblib.load(path)
            self.models[version] = bundle["model"]
            if self.columns is None:
                self.columns = bundle["columns"]

    def preprocess(self, features: dict):
        """
        Преобразует JSON в DataFrame
        """
        df = pd.DataFrame([features])
        return df[self.columns]
    
    def predict(self, features: dict, version: str):
        """
        Основной метод инференса
        """
        X = self.preprocess(features)
        model = self.models[version]
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        
        return {
            "prediction": int(pred),
            "probability": float(proba)
        }