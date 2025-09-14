import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os

class LinearRegressionGD:
    def __init__(self, learning_rate=0.001, epochs=1000, verbose=True, print_every=100, log_file="training_log.csv"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.print_every = print_every
        self.log_file = log_file
        self.m = 0.0
        self.c = 0.0
        self.loss_history = []
        self.training_log = []  # store iteration logs

    def update_parameters(self, X, Y, Y_pred, n):

        # MSE = 1/n (∑(y - y^)^2)
        """Compute gradients and update slope (m) and intercept (c)."""
        error = Y_pred - Y
        D_m = (2/n) * np.sum(X * error)  #∂MSE/∂m=2/n(∑X(y^−y))
        D_c = (2/n) * np.sum(error)      # ∂MSE/∂c=2/n(∑(y^−y))

        # new_parameter = old_parameter - learning_rate * loss_wrt_parameter
        self.m -= self.learning_rate * D_m 
        self.c -= self.learning_rate * D_c

        return error

    def fit(self, X, Y, save_gif=False, gif_name="regression_training.gif", save_every=10):
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        n = float(len(X))

        filenames = []
        if save_gif:
            if not os.path.exists("frames"):
                os.makedirs("frames")

        for i in range(self.epochs):
            Y_pred = self.m * X + self.c
            error = self.update_parameters(X, Y, Y_pred, n)

            # Loss calculation
            loss = np.mean(error**2)
            self.loss_history.append(loss)

            # Log activity
            self.training_log.append({
                "iteration": i+1,
                "loss": loss,
                "m": self.m,
                "c": self.c
            })

            # Print progress
            if self.verbose and (i % self.print_every == 0 or i == self.epochs - 1):
                print(f"Iteration {i+1}/{self.epochs}, Loss = {loss:.4f}, m = {self.m:.4f}, c = {self.c:.4f}")

            # Save frames for GIF
            if save_gif and (i % save_every == 0 or i == self.epochs-1):
                plt.figure()
                plt.scatter(X, Y, color="blue", label="Actual data")
                plt.plot(X, self.m*X + self.c, color="red", label=f"Iter {i+1}")
                plt.xlabel("Hours Studied")
                plt.ylabel("Scores")
                plt.legend()
                filename = f"frames/frame_{i+1}.png"
                plt.savefig(filename)
                plt.close()
                filenames.append(filename)

        # Save GIF
        if save_gif:
            images = [imageio.imread(f) for f in filenames]
            imageio.mimsave(gif_name, images, duration=0.3)
            print(f"GIF saved as {gif_name}")

            # Log gif path
            self.training_log.append({"iteration": "N/A", "loss": "N/A", "m": "N/A", "c": "N/A", "gif_path": gif_name})

        # Save log file
        pd.DataFrame(self.training_log).to_csv(self.log_file, index=False)
        print(f"Training log saved as {self.log_file}")

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        return self.m * X + self.c

    def plot_regression_line(self, X, Y):
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        plt.scatter(X, Y, color="blue", label="Actual data")
        plt.plot(X, self.predict(X), color="red", label="Fitted line")
        plt.xlabel("Hours Studied")
        plt.ylabel("Scores")
        plt.legend()
        plt.show()

    def plot_loss_curve(self):
        plt.plot(range(1, len(self.loss_history)+1), self.loss_history, color="green")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.title("Loss Curve (Convergence)")
        plt.show()


# Load dataset
dataset = pd.read_csv("student_score.csv")
X = dataset["Hours"]
Y = dataset["Scores"]

# Train model with logging
model = LinearRegressionGD(learning_rate=0.001, epochs=500, verbose=True, print_every=50, log_file="training_log.csv")
model.fit(X, Y, save_gif=True, gif_name="training.gif", save_every=25)

# Plot regression line
model.plot_regression_line(X, Y)

# Plot loss curve
model.plot_loss_curve()
print("slope: %.3f" %model.m)
print("intercept: %.3f" %model.c)

# Make prediction
hours = 9.25
predicted_score = model.predict([hours])
print(f"Predicted Score for {hours} study hours = {predicted_score[0][0]:.2f}")
