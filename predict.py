import mlflow.sklearn
import pandas as pd

model = mlflow.sklearn.load_model("models:/credit_model/1")

sample = pd.DataFrame({
    "Age": [30],
    "Income": [55000],
    "CreditScore": [680]
})

prediction = model.predict(sample)
print("✅ Prediction:", "Approved" if prediction[0] == 1 else "Rejected")
