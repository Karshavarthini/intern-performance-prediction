# Sprint 2: Classification + Regression Models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor

# Load Dataset
df = pd.read_csv("C:\\Users\\karsh\\OneDrive\\Desktop\\intern-performance-prediction\\intern_dataset (1).csv")

X = df.drop("performance_label", axis=1)
y = df["performance_label"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. CLASSIFICATION MODEL (XGBoost)
print("\n===== XGBOOST CLASSIFIER =====")

clf = XGBClassifier(eval_metric='mlogloss')
clf.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred_clf))

# 2. REGRESSION MODEL (Gradient Boosting)
print("\n===== GRADIENT BOOSTING REGRESSOR =====")

reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

# Convert regression output to class (0,1,2)
y_pred_reg_rounded = np.round(y_pred_reg)

print("Regression MSE:", mean_squared_error(y_test, y_pred_reg))
print("Regression Accuracy (after rounding):", accuracy_score(y_test, y_pred_reg_rounded))

print("\n===== FINAL COMPARISON =====")
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_clf))
print("Regression (Rounded) Accuracy:", accuracy_score(y_test, y_pred_reg_rounded))




