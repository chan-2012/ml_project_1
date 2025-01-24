import pytest
from app.main import train_model, predict_price
import os

def test_train_model():
    """Tests the model training function."""
    model = train_model()
    assert model is not None

def test_predict_price():
    """Tests the prediction function."""
    # Ensure the model file exists before testing prediction
    if not os.path.exists('app/model.joblib'):
        train_model()
    
    predicted_price = predict_price(1700)
    assert predicted_price is not None
    assert predicted_price > 0
