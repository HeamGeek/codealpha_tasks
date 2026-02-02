# ----------------------------------------
# Iris Flower Classification
# CodeAlpha Internship - Task 1
# ----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/Iris.csv")

print("Dataset Preview:")
print(data.head())

# Drop unnecessary columns
if "Id" in data.columns:
    data.drop(columns=["Id"], inplace=True)

# Encode target labels 
label_encoder = LabelEncoder()
data["Species"] = label_encoder.fit_transform(data["Species"])

# Split features and target
X = data.drop("Species", axis=1)
y = data["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Iris Classification")
plt.tight_layout()
plt.show()
