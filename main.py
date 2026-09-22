from src.evaluate import evaluate_lr
from src.train import training_lr
from src.train import training_rf
from src.train import training_xgboost
from src.evaluate import evaluate_rf
from src.evaluate import evaluate_xg

import os
import joblib



reg, X_test, y_test = training_rf('/Users/apple/Developer/ML/bangalore-house-price-prediction/datasets/dataset.csv')

os.makedirs('models', exist_ok=True)
joblib.dump(reg, 'models/house_prediction_model.pkl')
print("Model saved.")

evaluate_rf(reg, X_test, y_test)
