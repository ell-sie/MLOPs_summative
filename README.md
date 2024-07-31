# Machine Learning Pipeline Deployment with FastAPI

## Overview

This project demonstrates the deployment of a machine learning pipeline using FastAPI. The pipeline includes training a Multi-layer Perceptron (MLP) model, evaluating it, deploying it on a cloud platform, and monitoring its performance. The project also showcases how to retrain the model when needed and handle user interactions for predictions.

## Directory Structure

MLOPs_summative/
│
├── README.md
│
├── notebook/
│ └── project_name.ipynb
│
├── src/
│ ├── app.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── save_datasets.py
│
├── data/
│ ├── train/
│ │ └── iris_train.csv
│ └── test/
│ └── iris_test.csv
│
└── models/
├── mlp_model.pkl
└── scaler.pkl

## Setup

### Prerequisites

- Python 3.7+
- pip (Python package installer)
- Jupyter Notebook
- FastAPI
- Uvicorn

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ell-sie/MLOPs_summative.git
    cd MLOPs_summative
    ```

3. **Install the Required Packages**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Jupyter Notebook

1. **Navigate to the Notebook Directory**

    ```bash
    cd notebook
    ```

2. **Start Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

3. Open `iris.ipynb` and run all cells to train and save the model.


Sure, I'll update your README file to include the necessary instructions and add the Swagger UI endpoint. Here's the updated README.md:

markdown
Copy code
# MLOps Summative Project

This project demonstrates the deployment of a machine learning model using FastAPI. The model is a Multi-layer Perceptron (MLP) trained on the Iris dataset.

## Project Structure

project_name/
│
├── README.md
│
├── notebook/
│ └── project_name.ipynb
│
├── src/
│ ├── app.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── save_datasets.py
│ └── prediction.py
│
├── data/
│ ├── train/
│ │ └── iris_train.csv
│ └── test/
│ └── iris_test.csv
│
└── models/
├── mlp_model.pkl
└── scaler.pkl

markdown
Copy code

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/project_name.git
   cd project_name

## Directory Setup
Ensure the following directory structure:

```bash
data/
├── train/
│   └── iris_train.csv
└── test/
    └── iris_test.csv
```
## Prepare Data
Run the save_datasets.py script to save the Iris dataset into the train and test directories:

```bash
python src/save_datasets.py
```

## Train the Model
Run the model.py script to train the MLP model and save it along with the scaler:

```bash
python src/model.py
```

### Running the FastAPI Application

1. **Navigate to the Source Directory**

    ```bash
    cd src
    ```

2. **Start the FastAPI Application**

    ```bash
    uvicorn prediction:app --reload
    ```

3. Access the FastAPI application at `http://127.0.0.1:8000`.

## API Documentation

 You can access the interactive API documentation at http://127.0.0.1:8000/docs.

## Retraining

1. **Making Predictions**

You can make predictions by sending a POST request to the /predict endpoint.

Example Request with curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```
## Example Request with Postman

1. Set the request type to POST.

2. Set the URL to http://127.0.0.1:8000/predict.

3. In the Body tab, select raw and JSON format.

4. Enter the following JSON data:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
5. Click "Send" to get the prediction response.

## Model Files
The trained model and scaler are saved in the models directory:

1. mlp_model.pkl: The trained MLP model.
2. scaler.pkl: The scaler used to preprocess the data.

## Author
Elsie Ndiramiye

