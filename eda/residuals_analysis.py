# Observe the results from Linear Regression

# Import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from helpers_ml import *

#Load datasets
tasks_df = pd.read_csv('clean_tasks.csv')
suppliers_df = pd.read_csv('clean_suppliers.csv')
cost_df = pd.read_csv('clean_costs.csv')

# Merge all three datasets
final_df = combine_datasets(cost_df, tasks_df, suppliers_df)

# Separate into X, y, and Task groups
X, y, Groups = seperate_into_X_y(final_df)

# Set the random seed for reproducability
np.random.seed(42)

# Split into train and test sets
X_train, y_train, X_test, y_test, TaskGroup = train_test_split(X, y, Groups)

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Calculate residuals (difference between actual and predicted values)
residuals = y_test - y_pred

# Set the style
plt.style.use('ggplot')

# Plot the residuals to check normality
plt.figure(figsize=(10, 6))

# Histogram of residuals
plt.hist(residuals, color='blue', bins=30, edgecolor='black', alpha=0.7)
plt.title("Residuals Distribution", fontsize=16)
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# Residuals vs Predicted Values Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.7, edgecolor='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title("Residuals vs Predicted Values", fontsize=16)
plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.show()

# Scatter plot of actual data points
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, edgecolor='black', label="Predicted vs Actual")
plt.plot(y_test, y_test, color='red', linestyle='--', label="Perfect Prediction Line")
plt.title("Linear Regression Model: Actual vs Predicted", fontsize=16)
plt.xlabel("Actual Values (y_test)", fontsize=12)
plt.ylabel("Predicted Values (y_pred)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Review the equation
# Extract the intercept and coefficients 
intercept = lr_model.intercept_
coefficients = lr_model.coef_

# Create the regression equation as a string
feature_names = X.columns
equation = f"y = {intercept:.4f}"
for coef, feature in zip(coefficients, feature_names):
    equation += f" + ({coef:.4f})*{feature}"

# Print the equation
print("Linear Regression Equation:")
print(equation)


