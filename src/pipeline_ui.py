# src/pipeline_ui.py

import joblib
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import StringIO

app = FastAPI()

class IrisData(BaseModel):
    features: List[float]

@app.post("/predict/")
def predict(data: IrisData):
    scaler = joblib.load('../models/scaler.pkl')
    model = joblib.load('../models/mlp_model.pkl')
    
    X_new = np.array(data.features).reshape(1, -1)
    X_new = scaler.transform(X_new)
    
    prediction = model.predict(X_new)
    return {"prediction": int(prediction[0])}

@app.post("/retrain/")
def retrain(file: UploadFile = File(...)):
    data = pd.read_csv(StringIO(file.file.read().decode('utf-8')))
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    scaler = joblib.load('../models/scaler.pkl')
    X = scaler.transform(X)
    
    model = joblib.load('../models/mlp_model.pkl')
    model.partial_fit(X, y)
    
    joblib.dump(model, '../models/mlp_model.pkl')
    return {"status": "Model retrained successfully"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris MLP Model API"}
