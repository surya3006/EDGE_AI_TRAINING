import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import imageio, os
from IPython.display import Image, display

# ------------------ Load Dataset ------------------
dataset = pd.read_csv("C:\\Users\\startup_data.csv")

# Separate features & target
X = dataset[["R&D Spend", "Administration", "Marketing Spend"]]  # train on all features
Y = dataset["Profit"]

# ------------------ Train Model ------------------
model = LinearRegression()
model.fit(X, Y)

print("\n✅ Model Trained Successfully")
print("Coefficients (Weights):", model.coef_)
print("Intercept (Bias):", model.intercept_)

# ------------------ Predictions ------------------
Y_pred = model.predict(X)
df_compare = pd.DataFrame({"Actual Profit": Y, "Predicted Profit": Y_pred})
print("\n🔎 Actual vs Predicted Profit:")
print(df_compare.head(10))  # print first 10 rows

# ------------------ Save Actual vs Predicted ------------------

csv_path = "C:\\Users\\EDGE_AI\\Startup_polynomial_regression\\3D_actual_vs_predicted_profit.csv"
df_compare.to_csv(csv_path, index=False)

# ------------------ Prepare for GIF ------------------
steps = 30
frames = []

# Fix Administration at its mean
admin_mean = X["Administration"].mean()

X1 = X["R&D Spend"].values
X2 = X["Marketing Spend"].values

# Create meshgrid for plane
x1_range = np.linspace(X1.min(), X1.max(), 20)
x2_range = np.linspace(X2.min(), X2.max(), 20)
xx1, xx2 = np.meshgrid(x1_range, x2_range)

final_weights = model.coef_
final_bias = model.intercept_

for step in range(steps + 1):
    alpha = step / steps
    w = final_weights * alpha
    b = final_bias * alpha

    # Predict Z for meshgrid using all 3 features
    Z = w[0] * xx1 + w[1] * admin_mean + w[2] * xx2 + b

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X1, X2, Y, color="blue", label="Actual data", alpha=0.6)
    ax.plot_surface(xx1, xx2, Z, color="red", alpha=0.5)

    ax.set_xlabel("R&D Spend")
    ax.set_ylabel("Marketing Spend")
    ax.set_zlabel("Profit")
    ax.set_title(f"Regression Plane (Step {step}/{steps})")
    ax.view_init(elev=20, azim=40)

    plt.tight_layout()
    plt.savefig("frame.png")
    plt.close()
    frames.append(imageio.imread("frame.png"))

gif_path = "multi_regression_plane.gif"
imageio.mimsave(gif_path, frames, fps=5)

print(f"\n🎥 GIF saved as: {os.path.abspath(gif_path)}")

# ------------------ Display GIF ------------------
try:
    display(Image(filename=gif_path))
except:
    print("Open the file manually to view the GIF.")
