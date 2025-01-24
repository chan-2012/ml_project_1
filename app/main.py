import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import mlflow

def train_model(test_size=0.2, random_state=42):
    """Trains a linear regression model and logs metrics to MLflow."""
    mlflow.start_run() # Start an MLflow run

    data = pd.read_csv("data/house_prices.csv")
    X = data[["size"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    # Log metrics
    mlflow.log_metric("mse", mse)

    # Save the model
    joblib.dump(model, 'app/model.joblib')

    # Log the model artifact
    mlflow.sklearn.log_model(model, "model")

    mlflow.end_run() # End the MLflow run

    return model

def predict_price(size):
    """Predicts the house price for a given size."""
    try:
        model = joblib.load('app/model.joblib')
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return None

    return model.predict([[size]])[0]

if __name__ == "__main__":
    # Experiment with different parameters
    train_model(test_size=0.2, random_state=42)
    train_model(test_size=0.3, random_state=100)

    print(f"Predicted price for 1800 sq ft: {predict_price(1800)}")
