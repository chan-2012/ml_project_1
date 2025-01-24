import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    """Trains a linear regression model."""
    data = pd.read_csv("data/house_prices.csv")
    X = data[["size"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'app/model.joblib')

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
    train_model()
    print(f"Predicted price for 1800 sq ft: {predict_price(1800)}")
