import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(data):
    # Assuming 'data' is a pandas DataFrame
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, 'C:/Users/LENOVO/Desktop/Elsie files/MLOPs_summative/models/scaler.pkl')
    return X_scaled, y

def load_and_preprocess_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)
    
    return X_train, X_test, y_train, y_test
