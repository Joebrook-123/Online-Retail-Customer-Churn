import pandas as pd
from train import train_model
from transform import train_test_split, transform

df = pd.read_csv('online_retail_cleaned.csv', index_col=False)

X_train, y_train, X = train_test_split(df)
X = transform(X)    

model = train_model(X_train, y_train)

y_proba = model.predict_proba(X)[:, 1]

df['churn_probability'] = y_proba.ravel()
df.to_csv('customer_churn_predictions.csv', index=False)
