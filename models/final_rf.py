from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from helpers_ml import *

# Load datasets
tasks_df = pd.read_csv('clean_tasks.csv')
suppliers_df = pd.read_csv('clean_suppliers.csv')
cost_df = pd.read_csv('clean_costs.csv')

# Merge all three datasets
final_df = combine_datasets(cost_df, tasks_df, suppliers_df)
# Separate into X, y, and Task groups
X, y, Groups = seperate_into_X_y(final_df)
X_train, y_train, X_test, y_test, TaskGroup = train_test_split(X, y, Groups)


print('\nRandom Forest\n')

# Train the Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  
rf_model.fit(X_train, y_train)

# Predict Costs for Test Data
y_pred = rf_model.predict(X_test)

# Compute the R-squared
print("\nR2 Score:",rf_model.score(X_test, y_test))

#3.4 -- change 
results_df = pd.DataFrame({
    'Task ID': Groups[X_test.index],
    'Supplier ID': final_df.loc[X_test.index, 'Supplier ID'].values,
    'Actual Cost': y_test.values,
    'Predicted Cost': y_pred
})

final_results = calculate_errors(results_df)
errors = final_results['Error']
rmse = calculate_rmse(errors)
print(f'The calculated score(rmse) for RF model: {rmse}')

print("\nError for each task in TaskGroup\n", errors)
print("\nRMSE:", calculate_rmse(errors))

#4
print('\nLOGO')
logo = LeaveOneGroupOut()
error_scorer = make_scorer(error_func)
rf_error = cross_val_score(rf_model, X, y, cv=logo, groups=Groups, scoring=error_scorer)
print("RMSE cross-validation RF:", calculate_rmse(rf_error))

# Define the parameter grid
param_grid_rf = {
    'max_depth': [5,7,10],
    'n_estimators': [50, 100, 200]
}


# Set up GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    scoring=error_scorer,
    cv=logo.split(X, y, groups=Groups),  # Use Leave-One-Group-Out CV
    verbose=2,
    n_jobs=-1  # Use all available CPUs
)

# Run the Grid Search
grid_search_rf.fit(X, y)

# Best parameters and score
print("Best Parameters:", grid_search_rf.best_params_)
print("Best Score:", grid_search_rf.best_score_)

cv_results_rf = pd.DataFrame(grid_search_rf.cv_results_)
gs_error_rf = cv_results_rf.iloc[grid_search_rf.best_index_].iloc[10:-3]

rmse_rf = calculate_rmse(gs_error_rf)
print('RMSE score: ', rmse_rf)


#Best Parameters: {'max_depth': 10, 'max_features': 30, 'min_samples_leaf': 4, 'n_estimators': 200}
#Best Score: -0.01887979046455084
#RMSE score:  0.025384669367564828