import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Путь к данным
DATA_PATH = "./data/UCI_Credit_Card.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Удалим ID
    df = df.drop(columns=["ID"])

    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]

    return X, y


def train_and_save(version, model):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    print(classification_report(y_test, preds))

    joblib.dump({
        "model": pipeline,
        "columns": X.columns.tolist()
    }, f"./models/model_{version}.pkl")
    
    print(f"Model saved to ./models/model_{version}.pkl")


if __name__ == "__main__":
    # v1 
    train_and_save("v1", LogisticRegression(max_iter=1000))
    # v2
    train_and_save(
        "v2",
        RandomForestClassifier(
            max_depth=5, 
            random_state=42
        )
    )