from src.evaluate import evaluate_lr
from src.train import training_lr

import os
import joblib



reg, X_test, y_test = training_lr('/Users/apple/Developer/ML/bangalore-house-price-prediction/datasets/dataset.csv')

os.makedirs('models', exist_ok=True)
joblib.dump(reg, 'models/house_prediction_model.pkl')
print("Model saved.")

evaluate_lr(reg, X_test, y_test)
