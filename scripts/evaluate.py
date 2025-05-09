import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/voice_data.csv")
X = df.drop("gender", axis=1)
y = df["gender"]

model = joblib.load("models/voice_model.pkl")
y_pred = model.predict(X)
print(classification_report(y, y_pred))