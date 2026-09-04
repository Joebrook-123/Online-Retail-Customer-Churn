import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Split Dataframe into train and test
def train_test_split(df):
    df["snapshot"] = pd.to_datetime(df["snapshot"])
    df_srt = df.sort_values(["customer_id", "snapshot"])

    rev_rank = df_srt.groupby("customer_id").cumcount(ascending=False)

    train = df_srt[rev_rank >= 1]

    features = [
        "total_order_value",
        "total_orders_placed",
        "historic_avg_order_value",
        "total_discounts",
        "total_orders_returned",
        "unique_products_ordered",
        "total_discount_value",
        "total_returns_value",
        "snapshot_order_value",
        "snapshot_orders_placed",
        "snapshot_avg_order_value",
        "snapshot_total_discounts",
        "snapshot_orders_returned",
        "snapshot_unique_products_ordered",
        "snapshot_discount_value",
        "snapshot_returns_value",
        "customer_age"
    ]

    # Split data into test and train
    X_train = train[features].to_numpy()
    y_train = train['customer_churned']


    X = df[features].to_numpy()

    return X_train, y_train, X

# Create polynomial features and scale variables
def transform(X):
    poly = PolynomialFeatures(degree=3)
    scaler = StandardScaler()

    X = poly.fit_transform(X)
    X = scaler.fit_transform(X)

    return X
