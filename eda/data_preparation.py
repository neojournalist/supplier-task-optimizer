## Data loading and preparation

# Import packages
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from helpers_prep import *

# Load datasets
tasks_df = pd.read_csv('tasks.csv')
suppliers_df = pd.read_csv('suppliers.csv')
cost_df = pd.read_csv('cost.csv')

# Create names and list of dfs for the function
list_df = [tasks_df, suppliers_df, cost_df]
names = ['tasks', 'suppliers', 'cost']

# Overview of the datsets
get_overview_dfs(names, list_df)

# Clean datasets
clean_tasks = clean_dataset('tasks', tasks_df)
clean_suppliers = clean_dataset('suppliers', suppliers_df)
clean_suppliers = col_transpose(suppliers_df, 'Features')

# Reset indexes
clean_tasks = clean_tasks.set_index('Task ID')
clean_suppliers.index.name = 'Supplier ID'
cost_df = cost_df.set_index('Task ID')

# Preprocess datasets
proc_tasks = preprocess_data(clean_tasks, 'tasks')
proc_suppliers = preprocess_data(clean_suppliers, 'suppliers')
print(f'The rows and columns in tasks: {proc_tasks.shape}')
print(f'The rows and columns in suppliers: {proc_suppliers.shape}')

filtered_tasks = filter_not_in_costs(proc_tasks, cost_df)
print(f'There are {filtered_tasks.shape} task IDs in tasks after cost filtering.')

# Drop suppliers with even higher costs according to Error function
final_cost, final_suppliers = iqr_drop_suppliers(cost_df, proc_suppliers)
final_cost, final_suppliers = high_error_supp_drop(final_cost, final_suppliers)

# Save generated dfs to csv for EDA ---> uncomment for saving
filtered_tasks.to_csv('clean_tasks.csv')
final_suppliers.to_csv('clean_suppliers.csv')
final_cost.to_csv('clean_costs.csv')


