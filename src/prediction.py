import numpy as np
import joblib

def make_prediction(features):
    scaler = joblib.load('./models/scaler.pkl')
    model = joblib.load('./models/mlp_model.pkl')
    
    X_new = np.array(features).reshape(1, -1)
    X_new = scaler.transform(X_new)
    
    prediction = model.predict(X_new)
    return int(prediction[0])
