from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
import numpy as np


def evaluate_lr(reg, X_test, y_test):

    y_pred = reg.predict(X_test)
    y_pred_rupees = np.expm1(y_pred)
    y_test_ruppes = np.expm1(y_test)
    print(r2_score(y_pred_rupees, y_test_ruppes))

def evaluate_rf(reg, X_test, y_test):

    y_pred = reg.predict(X_test)
    y_pred_rupees = np.expm1(y_pred)
    y_test_ruppes = np.expm1(y_test)
    print(r2_score(y_test_ruppes, y_pred_rupees))

def evaluate_xg(reg, X_test, y_test):

    y_pred = reg.predict(X_test)
    y_pred_rupees = np.expm1(y_pred)
    y_test_ruppes = np.expm1(y_test)
    print(r2_score(y_test_ruppes, y_pred_rupees))