import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Путь к данным
DATA_PATH = "/Users/andreygavrilov/Documents/Projects/credit-card-ml-deployment/data/UCI_Credit_Card.csv"
MODEL_PATH = "./models/model_v1.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Удалим ID
    df = df.drop(columns=["ID"])

    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]

    return X, y


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    print(classification_report(y_test, preds))

    joblib.dump({
        "model": pipeline,
        "columns": X.columns.tolist()
    }, MODEL_PATH)
    
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()