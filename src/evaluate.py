from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd


def evaluate_lr(reg, X_test, y_test):

    y_pred = reg.predict(X_test)
    y_pred_rupees = np.expm1(y_pred)
    y_test_ruppes = np.expm1(y_test)
    print(r2_score(y_pred_rupees, y_test_ruppes))
    print(root_mean_squared_error(y_test_ruppes, y_pred_rupees))
    mape = np.mean(np.abs((y_test_ruppes - y_pred_rupees) / y_test_ruppes)) * 100
    print(f"MAPE: {mape:.2f}%")

def evaluate_rf(reg, X_test, y_test):

    y_pred = reg.predict(X_test)
    y_pred_rupees = np.expm1(y_pred)
    y_test_ruppes = np.expm1(y_test)
    print(r2_score(y_test_ruppes, y_pred_rupees))
    print(root_mean_squared_error(y_test_ruppes, y_pred_rupees))
    mape = np.mean(np.abs((y_test_ruppes - y_pred_rupees) / y_test_ruppes)) * 100
    print(f"MAPE: {mape:.2f}%")

def evaluate_xg(reg, X_test, y_test):

    y_pred = reg.predict(X_test)
    y_pred_rupees = np.expm1(y_pred)
    y_test_ruppes = np.expm1(y_test)
    print(r2_score(y_test_ruppes, y_pred_rupees))
    mape = np.mean(np.abs((y_test_ruppes - y_pred_rupees) / y_test_ruppes)) * 100
    print(f"MAPE: {mape:.2f}%")