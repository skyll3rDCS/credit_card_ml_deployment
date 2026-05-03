import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "./models/model_v1.pkl"


class ModelHandler:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        data = joblib.load(MODEL_PATH)
        self.columns = data["columns"]
        return data["model"]

    def preprocess(self, input_data: dict):
        """
        Преобразует JSON в DataFrame
        """
        df = pd.DataFrame([input_data])
        df = df[self.columns]  # гарантируем порядок
        return df
    
    def predict(self, input_data: dict):
        """
        Основной метод инференса
        """
        processed_data = self.preprocess(input_data)

        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }