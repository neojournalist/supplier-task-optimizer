import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

"""
The functions are used in model training, fitting and assesing part.
The first three functions are used to combine tasks, suppliers and cost datasets to generate one and separate into training and testing sets based on task groups.
The next functions help to calculate errors of minimum cost and predicted supplier cost, and rmse.
The final the error function is the custom score function to pass for cross validation parameters.
"""

# Combine all three datasets into one on IDs
def combine_datasets(cost_df, tasks_df, suppliers_df):
    task_cost_df = pd.merge(cost_df, tasks_df, on="Task ID", how="left")
    final_df = pd.merge(task_cost_df, suppliers_df, left_on="Supplier ID", right_on="Supplier ID", how="left")
    
    # Drop unnecessary columns
    #final_df.drop(columns=["Unnamed: 0"], inplace=True)

    if len(cost_df) != len(final_df):
        print("Warning: The number of rows in the final DataFrame does not match the costs dataset.")
    
    return final_df
    


# Separate into features and target + groups of Task IDs
def seperate_into_X_y(final_df):
    X = final_df.drop(['Task ID', 'Supplier ID', 'Cost'], axis=1)
    y = final_df['Cost']
    Groups = final_df['Task ID']
    
    return X, y, Groups

# Split the dataset into X_train and test with 20 random Task IDs
def train_test_split(X, y, Groups, test_size=20):
    tasks_id = Groups.unique()
    TaskGroup = np.random.choice(tasks_id, size=test_size, replace=False)

    # split the dataset into training and testing sets
    random_task = Groups.isin(TaskGroup)
    X_train = X[~random_task]
    X_test = X[random_task]
    y_train = y[~random_task]
    y_test = y[random_task]


    return X_train, y_train, X_test, y_test, TaskGroup

#Scoring function to calculate error between true min cost among all suppliers per task and  and predicted supplier
def calculate_errors(results_df):
    min_cost_per_task = results_df.groupby('Task ID')['Actual Cost'].min().rename('Minimum Cost Per Task')
    results_with_min_cost = results_df.join(min_cost_per_task, on='Task ID')

    # Find rows with minimum predicted cost for each task
    min_predicted_cost_rows = results_with_min_cost.loc[
        results_with_min_cost.groupby('Task ID')['Predicted Cost'].idxmin()
    ]
    min_predicted_cost_rows = min_predicted_cost_rows.copy()
    # Add error column directly while creating the final DataFrame
    min_predicted_cost_rows['Error'] = min_predicted_cost_rows['Minimum Cost Per Task'] - min_predicted_cost_rows['Actual Cost']

    # Compute the error
    return min_predicted_cost_rows

# Calculate rmse passing errors
def calculate_rmse(errors):
    rmse = np.sqrt(np.mean(errors**2))
    return rmse

# Custom error function to pass for make_scorer
def error_func(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    min_actual_cost = np.min(y_test)
    best_supplier_index = np.argmin(y_pred)
    best_supplier_actual_cost = y_test[best_supplier_index]
    
    return min_actual_cost - best_supplier_actual_cost


