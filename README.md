# Machine Learning Pipeline Deployment with FastAPI

## Overview

This project demonstrates the deployment of a machine learning pipeline using FastAPI. The pipeline includes training a Multi-layer Perceptron (MLP) model, evaluating it, deploying it on a cloud platform, and monitoring its performance. The project also showcases how to retrain the model when needed and handle user interactions for predictions.

## Directory Structure

Project_name/
│
├── README.md
│
├── notebook/
│ ├── project_name.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ └── prediction.py
│
├── data/
│ ├── train/
│ └── test/
│
└── models/
├── iris_model.pkl
└── iris_model.tf

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
    git clone https://github.com/yourusername/Project_name.git
    cd Project_name
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
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

3. Open `project_name.ipynb` and run all cells to train and save the model.

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

## Deployment

### Docker Deployment

1. **Build the Docker Image**

    ```bash
    docker build -t project_name .
    ```

2. **Run the Docker Container**

    ```bash
    docker run -d -p 8000:8000 project_name
    ```

### Cloud Platform Deployment

Instructions for deploying the FastAPI application on your chosen cloud platform (e.g., AWS, GCP, Azure) will be added here.

## Retraining

Instructions for setting up retraining triggers on the cloud platform will be added here.

## Load Testing

### Using Locust

1. **Install Locust**

    ```bash
    pip install locust
    ```

2. **Create a Locustfile**

    Instructions for creating a locustfile and running load tests will be added here.
