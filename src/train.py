import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from src.preprocess import load_and_preprocess


def training_lr(filepath):

    X, y, preprocessor = load_and_preprocess(filepath)
    y_log = np.log1p(y)

    reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    reg.fit(X_train, y_train)

    return reg, X_test, y_test

def training_rf(filepath):

    X, y, preprocessor = load_and_preprocess(filepath)
    y_log = np.log1p(y)
    
    reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=4, min_samples_split=2, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    reg.fit(X_train, y_train)

    return reg, X_test, y_test

def training_xgboost(filepath):

    X, y, preprocessor = load_and_preprocess(filepath)
    y_log = np.log1p(y)
    
    reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    reg.fit(X_train, y_train)

    return reg, X_test, y_test