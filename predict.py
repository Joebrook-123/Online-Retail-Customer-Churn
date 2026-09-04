import pandas as pd
from train import train_model
from transform import train_test_split, transform
import pantab

df = pd.read_csv('online_retail_cleaned.csv', index_col=False)
df['snapshot'] = pd.to_datetime(df['snapshot'])

X_train, y_train, X = train_test_split(df)
X = transform(X)    

model = train_model(X_train, y_train)
y_proba = model.predict_proba(X)[:, 1]
df['churn_probability'] = y_proba.ravel()

# Cross joining data for visualisation in Tableau
unique_customers = df['customer_id'].drop_duplicates()
unique_snapshots = df['snapshot'].drop_duplicates()

# Cross join grid
grid = pd.MultiIndex.from_product(
    [unique_customers, unique_snapshots],
    names=['customer_id', 'snapshot'],
).to_frame().reset_index(drop=True)

# Join original dataframe onto every snapshot and customer combination
df = pd.merge(
    grid, 
    df, 
    on=['customer_id', 'snapshot'],
    how='left'
)

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
df.drop(columns=['pct_rank'], inplace=True)

# Forward fill missing snapshots with the previous snapshots value for cumulative measures
df = df.sort_values(['customer_id', 'snapshot']).reset_index(drop=True)

cumulative_cols = [
    'total_discount_value',
    'total_discounts',
    'total_order_value',
    'total_orders_placed',
    'total_orders_returned',
    'total_returns_value'
]

df[cumulative_cols] = (
    df.groupby('customer_id')[cumulative_cols]
    .ffill()
    .fillna(0)
)

numeric_col = [
    'total_order_value',
    'total_orders_placed',
    'historic_avg_order_value',
    'total_discounts',
    'total_orders_returned',
    'unique_products_purchased',
    'total_discount_value',
    'total_returned_value',
    'snapshot_customer_count',
    'snapshot_orders_placed',
    'snapshot_avg_order_value',
    'snapshot_total_discounts',
    'snapshot_customer_orders',
    'snapshot_unique_products',
    'snapshot_customer_revenue',
    'snapshot_returned_value',
    'customer_churn_status',
    'churn_probability'
]

for col in numeric_col:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Hyper files do not support 32bit floats
float_cols = df.select_dtypes(include=['float32']).columns
df[float_cols] = df[float_cols].astype('float64')

# Export results to a Tableau hyper file
pantab.frame_to_hyper(df, 'customer_churn_predictions.hyper', table='churn_data')
print("Successfully generated Tableau Hyper extract!")
