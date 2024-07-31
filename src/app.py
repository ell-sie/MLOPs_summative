from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model and scaler
model_path = 'C:/Users/LENOVO/Desktop/Elsie files/MLOPs_summative/models/mlp_model.pkl'
scaler_path = 'C:/Users/LENOVO/Desktop/Elsie files/MLOPs_summative/models/scaler.pkl'
mlp = joblib.load(model_path)
scaler = joblib.load(scaler_path)

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris prediction API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    df = pd.DataFrame(data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    X_scaled = scaler.transform(df)
    prediction = mlp.predict(X_scaled)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
