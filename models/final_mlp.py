#Training MLP model

# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from helpers_ml import *
from helpers_viz import *

#Load datasets
tasks_df = pd.read_csv('clean_tasks.csv')
suppliers_df = pd.read_csv('clean_suppliers.csv')
cost_df = pd.read_csv('clean_costs.csv')

# Merge all three datasets
final_df = combine_datasets(cost_df, tasks_df, suppliers_df)

# Separate into X, y, and Task groups
X, y, Groups = seperate_into_X_y(final_df)

# Set the random seed for reprodusability
np.random.seed(42)

# Split into train and test sets
X_train, y_train, X_test, y_test, TaskGroup = train_test_split(X, y, Groups)

# Train the MLP Model
mlp = MLPRegressor(random_state=42, hidden_layer_sizes=(50,), solver='lbfgs', max_iter=1000)
mlp.fit(X_train, y_train)

# Predict Costs for Test Data
y_pred = mlp.predict(X_test)

# Compute the R-squared
print(f'R2 Score for the MLP model: {mlp.score(X_test, y_test)}')

# Create the initial DataFrame with relevant data
results_df = pd.DataFrame({
    'Task ID': Groups[X_test.index],
    'Supplier ID': final_df.loc[X_test.index, 'Supplier ID'].values,
    'Actual Cost': y_test.values,
    'Predicted Cost': y_pred
})

final_results = calculate_errors(results_df)
errors = final_results['Error']
rmse = calculate_rmse(errors)
print(f'The calculated score(rmse) for MLP model: {rmse}')

# Training MLP with cross validation LOGO

# Create the scorer with our own function
error_scorer = make_scorer(error_func)

# Set up Leave-One-Group-Out and cross-validation
logo = LeaveOneGroupOut()

mlp_error = cross_val_score(mlp, X, y, cv=logo, groups=Groups, scoring=error_scorer)


rmse = calculate_rmse(mlp_error)
print('RMSE score with cross-validation: ', rmse)

#5
# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [
        (50,), (100,), (50, 50), (100, 50), (100, 100), (50, 50, 50)
    ],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'activation': ['relu', 'tanh', 'logistic', 'identity']
}


# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring=error_scorer,
    cv=logo.split(X, y, groups=Groups),  # Use Leave-One-Group-Out CV
    verbose=2,
    n_jobs=-1  # Use all available CPUs
)

# Run the Grid Search
grid_search.fit(X, y)

# Best parameters and score
print("Best Parameters after Grid Search:", grid_search.best_params_)
print("Best Score - Grid Search:", grid_search.best_score_)

# Store the results in a dataframe
cv_results = pd.DataFrame(grid_search.cv_results_)

# Extract errors for the best feature model from Grid Search from all tasks
gs_error = cv_results.iloc[grid_search.best_index_].iloc[10:-3]

#Calculate rmse
rmse = calculate_rmse(gs_error)
print('RMSE score - Grid Search: ', rmse)