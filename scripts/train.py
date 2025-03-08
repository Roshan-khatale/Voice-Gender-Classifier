import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/voice_data.csv")
X = df.drop("gender", axis=1)
y = df["gender"]

model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "models/voice_model.pkl")
print("Model trained and saved.")