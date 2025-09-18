import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# ------------------ Load Data ------------------
dataset = pd.read_csv("C:\\Users\\startup_data.csv")

# Features and Target
X = dataset[["R&D Spend", "Administration", "Marketing Spend"]]  # All 3 features
Y = dataset["Profit"]

# ------------------ Train-Test Split ------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ------------------ Polynomial Transformation ------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

# Train Polynomial Regression Model
model = LinearRegression()
model.fit(X_train_poly, Y_train)

print("\n✅ Model Trained Successfully")
print("Number of polynomial features:", X_train_poly.shape[1])
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# ------------------ Partial Dependence Curve ------------------
# Fix R&D Spend & Administration at mean values
RD_mean = X["R&D Spend"].mean()
admin_mean = X["Administration"].mean()

# Generate range of Marketing Spend for curve
marketing_range = np.linspace(X["Marketing Spend"].min(), X["Marketing Spend"].max(), 100)

# Build input matrix with fixed values + varying Marketing Spend
X_curve = np.column_stack([
    np.full_like(marketing_range, RD_mean),   # R&D Spend fixed
    np.full_like(marketing_range, admin_mean),# Administration fixed
    marketing_range                           # Marketing Spend varies
])

# Transform to polynomial and predict
X_curve_poly = poly.transform(X_curve)
Y_curve_pred = model.predict(X_curve_poly)

# ------------------ Plot ------------------
plt.scatter(X["Marketing Spend"], Y, color="blue", label="Actual Data", alpha=0.6)
plt.plot(marketing_range, Y_curve_pred, color="red", linewidth=2, label="Polynomial Regression Curve")

plt.xlabel("Marketing Spend")
plt.ylabel("Predicted Profit")
plt.title("Partial Dependence Plot: Effect of Marketing Spend")
plt.legend()
plt.show()

# ------------------ Compare Actual vs Predicted (optional) ------------------
Y_train_pred = model.predict(X_train_poly)
df_compare = pd.DataFrame({"Actual": Y_train.values, "Predicted": Y_train_pred})
print("\n🔎 Actual vs Predicted (first 10 rows):")
print(df_compare.head(10))

# ------------------ Save Actual vs Predicted ------------------

csv_path = "C:\\Users\\EDGE_AI\\Startup_polynomial_regression\\actual_vs_predicted_profit.csv"
df_compare.to_csv(csv_path, index=False)
