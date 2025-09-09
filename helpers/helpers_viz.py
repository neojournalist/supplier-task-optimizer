# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


"""
The functions in the file are helpers to generate plots for EDA and for the error distribution.
"""

def cost_distribution(cost_df, title): 
    plt.figure(figsize=(10, 6))
    plt.hist(cost_df['Cost'], bins=30, edgecolor='black', alpha=0.7) 
    plt.title(title, fontsize=16)
    plt.xlabel('Cost', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

def tasks_distribution(tasks_df, title):
    plt.figure(figsize=(16, 10))
    tasks_df.boxplot()
    plt.grid(False)
    plt.title(title, fontsize=16)
    plt.ylabel('Task Feature Values')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Correlation matrix for a given dataset
def plot_correlation_matrix(data, dataset_name):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('Correlation Coefficient')

    plt.xticks(ticks=np.arange(len(data.columns)), labels=data.columns, rotation=90, fontsize=10)
    plt.yticks(ticks=np.arange(len(data.columns)), labels=data.columns, fontsize=10)
    plt.title(f"Correlation Matrix for {dataset_name} Features", fontsize=16)
    plt.tight_layout()
    plt.show()

# Heatmap to see the combination of suppliers and tasks
def heatmap_cost_tasks_suppliers(cost_df):
    cost_df_sorted = cost_df.copy()
    cost_df_sorted['Supplier ID'] = cost_df_sorted['Supplier ID'].apply(lambda x: int(re.search(r'\d+', x).group()))
    cost_df_sorted.sort_values(by='Supplier ID', inplace=True)
    cost_df_sorted['Supplier ID'] = 'S' + cost_df_sorted['Supplier ID'].astype(str)

    # A pivot table for cost patterns across suppliers and tasks
    cost_pivot = cost_df_sorted.pivot(index="Task ID", columns="Supplier ID", values="Cost")
    data = cost_pivot.values

    plt.figure(figsize=(16, 15))
    plt.imshow(data, cmap="coolwarm", aspect="auto")

    cbar = plt.colorbar()
    cbar.set_label("Cost")

    plt.xticks(ticks=np.arange(cost_pivot.columns.size), labels=cost_pivot.columns, rotation=90)
    plt.yticks(ticks=np.arange(cost_pivot.index.size), labels=cost_pivot.index)
    plt.xlabel("Supplier ID")
    plt.ylabel("Task ID")
    plt.title("Heatmap of Costs Across Tasks and Suppliers", fontsize=16)
    plt.tight_layout()
    plt.show()

# Calculate min_cost and rmse across suppliers
def calculate_err_and_rmse(cost_df):
    min_cost_per_task = cost_df.groupby('Task ID')['Cost'].min().reset_index()
    min_cost_per_task.rename(columns={'Cost': 'Min Cost'}, inplace=True)
    
    cost_full_min_cost = cost_df.merge(min_cost_per_task, on='Task ID', how='left')
    
    cost_full_min_cost['Error'] =  cost_full_min_cost['Min Cost'] - cost_full_min_cost['Cost']
    
    # Calculate RMSE
    cost_full_min_cost['Squared Error'] = cost_full_min_cost['Error'] ** 2
    sse_by_supplier = cost_full_min_cost.groupby('Supplier ID')['Squared Error'].sum()
    rmse_by_supplier = np.sqrt(sse_by_supplier / cost_full_min_cost['Task ID'].nunique())
    return cost_full_min_cost, rmse_by_supplier 

# Plot error distribution among suppliers
def error_distribution_by_supplier(cost_full_min_cost, cost_full_rmse_by_supplier):
    plt.figure(figsize=(16, 15))
    error_data_by_supplier = [
        cost_full_min_cost[cost_full_min_cost['Supplier ID'] == supplier]['Error']
        for supplier in cost_full_min_cost['Supplier ID'].unique()
    ]
    plt.boxplot(error_data_by_supplier, labels=cost_full_min_cost['Supplier ID'].unique())
    plt.title('Error Distribution Across Suppliers', fontsize=16)
    plt.xlabel('Supplier ID')
    plt.ylabel('Error')
    plt.xticks(rotation=90)
    
    # Add RMSE annotations 
    for i, supplier in enumerate(cost_full_min_cost['Supplier ID'].unique(), start=1):
        rmse_value = cost_full_rmse_by_supplier[supplier]
        plt.text(i, -0.245, f'{rmse_value:.4f}', ha='center', va='top', fontsize=10, color='blue', rotation=90)
    
    plt.subplots_adjust(bottom=0.3) 
    plt.tight_layout()
    plt.show()

# Plot error distribution across tasks
def error_distribution_by_task(cost_full_min_cost):
    # Extract median Error for each Task ID
    median_error_by_task = cost_full_min_cost.groupby('Task ID')['Error'].median().reset_index()
    median_error_by_task.rename(columns={'Error': 'Median Error'}, inplace=True)
    
    plt.figure(figsize=(16, 15))
    error_data_by_task = [
        cost_full_min_cost[cost_full_min_cost['Task ID'] == task]['Error']
        for task in cost_full_min_cost['Task ID'].unique()
    ]
    plt.boxplot(error_data_by_task, labels=cost_full_min_cost['Task ID'].unique())
    plt.title('Error Distribution Across Tasks', fontsize=16)
    plt.xlabel('Task ID')
    plt.ylabel('Error')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3) 
    plt.tight_layout()
    plt.show()
    
    # Return the median values as a DataFrame
    return median_error_by_task

# Plot of a single boxplot to see the distribution of errors
def plot_overall_median_error_boxplot(data, title):
    plt.figure(figsize=(6, 8))
    plt.boxplot(data, patch_artist=True,
                medianprops=dict(color='yellow'))
    
    plt.title(title, fontsize=16)
    plt.ylabel('Median Error', fontsize=12)
    plt.tight_layout()
    plt.show()















