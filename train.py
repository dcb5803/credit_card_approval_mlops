import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Embedded dataset
data = pd.DataFrame({
    "Age": [25, 40, 35, 28, 50],
    "Income": [50000, 62000, 58000, 52000, 80000],
    "CreditScore": [650, 700, 600, 720, 580],
    "Approved": [1, 1, 0, 1, 0]
})

X = data[["Age", "Income", "CreditScore"]]
y = data["Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    print(f"Logged model with accuracy: {acc}")
