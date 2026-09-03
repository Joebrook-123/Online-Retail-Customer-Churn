import pandas as pd
import numpy as np
import xgboost as xgb
from transform import transform

def train_model(X_train, y_train):
    X_train = transform(X_train)

    model = xgb.XGBClassifier(
        n_estimators = 100, 
        max_depth = 4,
        learning_rate = 0.1,
        random_state = 42,
    )

    model.fit(X_train, y_train)

    return model
