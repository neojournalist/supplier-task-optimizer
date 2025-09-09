from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from helpers import *

# Load datasets
tasks_df = pd.read_csv('clean_tasks.csv')
suppliers_df = pd.read_csv('clean_suppliers.csv')
cost_df = pd.read_csv('clean_costs.csv')

#3.2
final_df = combined_datasets(cost_df, tasks_df, suppliers_df)
X, y, Groups = seperate_into_X_y(final_df)
TestGroup, X_train, X_test, y_train, y_test = train_test_split(final_df, X, y, Groups)

#3.3
print('\nLasso\n')

# Train the Model
lasso = Lasso(alpha=0.0001) 
lasso.fit(X_train, y_train)

# Predict Costs for Test Data
y_pred = lasso.predict(X_test)

# Compute the R-squared
print("\nR2 Score:", lasso.score(X_test, y_test))

#3.4
error = calculate_error(final_df, Groups, X_test, y_test, y_pred)
print("\nError for each task in TaskGroup\n", error)
print("\nRMSE:", calculate_rmse(error))

#4
print('\nLOGO')
logo = LeaveOneGroupOut()
error_scorer = make_scorer(error_cv)
mlp_error = cross_val_score(lasso, X, y, cv=logo, groups=Groups, scoring=error_scorer)
print("RMSE:", calculate_rmse(mlp_error))

