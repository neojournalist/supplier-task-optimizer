## EDA
# Import packages
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

from helpers_viz import *

# Load datasets
tasks_df = pd.read_csv('clean_tasks.csv', index_col='Task ID')
suppliers_df = pd.read_csv('clean_suppliers.csv', index_col='Supplier ID')
cost_df = pd.read_csv('clean_costs.csv')

# Set the style
plt.style.use('ggplot')

# Plot correlation matrices of task features and supplier features
plot_correlation_matrix(tasks_df, "Tasks")
plot_correlation_matrix(suppliers_df, "Suppliers")

# Heatmap to identify combinations of tasks and suppliers
heatmap_cost_tasks_suppliers(cost_df)

# Calculate error Eq1: Error = Cost Min - Cost (supplier)
# Calculate error Eq2: RMSE
cost_distribution(cost_df, 'Distribution of Costs')
tasks_distribution(tasks_df, 'Distribution of Task Feature')
cost_full_min_cost, cost_full_rmse_by_supplier = calculate_err_and_rmse(cost_df)

# Plot error distribution
error_distribution_by_supplier(cost_full_min_cost, cost_full_rmse_by_supplier)

# error_distribution_by_task()
median_errors = error_distribution_by_task(cost_full_min_cost)
data = median_errors['Median Error']
plot_overall_median_error_boxplot(data, 'Boxplot of Median Errors Across All Tasks')