import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to DataFrame for easier handling
train_data = pd.DataFrame(X_train, columns=iris.feature_names)
train_data['target'] = y_train

test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data['target'] = y_test

# Create directories if they don't exist
train_dir = '../data/train'
test_dir = '../data/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

print(f"Directories created: {train_dir}, {test_dir}")

# Save the datasets as CSV files
train_file = os.path.join(train_dir, 'iris_train.csv')
test_file = os.path.join(test_dir, 'iris_test.csv')
train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

print(f"Files saved: {train_file}, {test_file}")

print("Datasets saved successfully.")
