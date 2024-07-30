# src/model.py

import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def train_mlp_model():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    
    # Save the trained model and scaler
    joblib.dump(mlp, 'C:/Users/LENOVO/Desktop/Elsie files/MLOPs_summative/models/mlp_model.pkl')
    joblib.dump(scaler, 'C:/Users/LENOVO/Desktop/Elsie files/MLOPs_summative/models/scaler.pkl')
    
    # Evaluate the model
    y_pred = mlp.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    train_mlp_model()
