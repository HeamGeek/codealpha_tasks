# ------------------------------------------
# Sales Prediction using Python
# CodeAlpha Internship – Task 4
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Load Dataset ----
df = pd.read_csv("data/advertising.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# ---- Exploratory Data Analysis ----
sns.pairplot(df)
plt.suptitle("Advertising Spend vs Sales", y=1.02)
plt.show()

# ---- Correlation Heatmap ----
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation between Advertising Channels and Sales")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png", dpi=150)
plt.show()

# ---- Feature & Target Selection ----
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Model Training ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Predictions ----
y_pred = model.predict(X_test)

# ---- Model Evaluation ----
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2 * 100:.2f}%")

# ---- Actual vs Predicted Plot ----
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("results/actual_vs_predicted_sales.png", dpi=150)
plt.show()

# ---- Feature Importance (Coefficients) ----
importance = pd.DataFrame({
    "Advertising Channel": X.columns,
    "Impact on Sales": model.coef_
})

print("\nAdvertising Impact on Sales:")
print(importance)

sns.barplot(
    x="Impact on Sales",
    y="Advertising Channel",
    data=importance
)
plt.title("Impact of Advertising Channels on Sales")
plt.tight_layout()
plt.savefig("results/advertising_impact.png", dpi=150)
plt.show()
