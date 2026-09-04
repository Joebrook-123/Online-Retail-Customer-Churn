import pandas as pd
from train import train_model
from transform import train_test_split, transform

df = pd.read_csv('online_retail_cleaned.csv', index_col=False)

X_train, y_train, X = train_test_split(df)
X = transform(X)    

model = train_model(X_train, y_train)

y_proba = model.predict_proba(X)[:, 1]

df['churn_probability'] = y_proba.ravel()

# Defining categories of risk based on percentiles
df['pct_rank'] = df.groupby('snapshot')['churn_probability'].rank(method='first', pct=True)

def map_risk(pct):
    if pct >= 0.70:
        return 'High Risk'
    elif pct >= 0.30:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['risk_category'] = df['pct_rank'].apply(map_risk)

# 3. Clean up the temporary column (optional)
df.drop(columns=['pct_rank'], inplace=True)

df.to_csv('customer_churn_predictions.csv', index=False)
