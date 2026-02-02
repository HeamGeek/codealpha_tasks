# ----------------------------------------
# Car Price Prediction with Machine Learning
# CodeAlpha Internship – Task 3
# ----------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ---- Load Dataset ----
data_path = os.path.join("data", "car data.csv")
df = pd.read_csv(data_path)

print("Dataset preview:")
print(df.head())

print("\nData info:")
print(df.info())

# ---- Handle Missing Values ----
print("\nMissing values before filling:")
print(df.isnull().sum())

df = df.dropna()  # drop rows with missing data

print("\nShape after dropping missing rows:", df.shape)

# Extract approximate brand goodwill
df["Brand"] = df["Car_Name"].str.split().str[0].str.lower()

# ---- Feature Selection ----
features = [
    "Year",             
    "Present_Price",    
    "Driven_kms",       
    "Fuel_Type",        
    "Selling_type",                     
    "Transmission",     
    "Owner",            
    "Brand"             
]

target = "Selling_Price"  # target variable

X = df[features]
y = df[target]

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Preprocessing ----

cat_cols = ["Fuel_Type", "Selling_type", "Transmission", "Owner", "Brand"]
num_cols = ["Year", "Present_Price", "Driven_kms"]

# one-hot encode  
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# ---- Model ----
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# ---- Train ----
model.fit(X_train, y_train)

# ---- Predict ----
y_pred = model.predict(X_test)

# ---- Evaluation Metrics ----
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2 * 100:.2f}%")

# ---- Plot True vs Predicted ----
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.savefig("results/car_price_actual_vs_predicted.png", dpi=150)
plt.show()
