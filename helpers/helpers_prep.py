import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
The following functions are created: 1) to give an overview of the datasets and 2) prepare for further analysis and model training.
The clean function cleans the format of observations in the dataset and drops entire columns with 0 values.
The preprocessing functions are built to preprocess data: drop with low variance,
drop highly correlated features, and scale datasets using MinMaxScaler.
The IQR function drops suppliers which are in the highest 25% of median cost. It's goal is to exclude worst performing suppliers based on median cost.
The high error supplier drop function excludes suppliers which have high RMSE compared to other suppliers and returns filtered cost and supplier DataFrames
"""

def get_overview_dfs(names, list_df):
    # Review datasets
    for name, df in zip(names, list_df):
        print(f'There are {df.shape[1]} columns in dataset: {name}.')
        print(f'There are {df.shape[0]} rows in dataset: {name}.')
    print('-'*60)
     # Check missing values
    for name, df in zip(names, list_df):
        missing = df.isnull().sum().sum()
        print(f'Missing values in {name}: {missing}.')
        print('\n')
    print('-'*60)
    # Display minimum and maximum values for each dataset
    for name, df in zip(names, list_df):
        numeric_df = df.select_dtypes(include=['number'])
        print(f'The minimum value of {name}: {np.min(numeric_df.values)}')
        print(f'The maximum value of {name}: {np.max(numeric_df.values)}')
    print('-'*30, 'end of overview', '-'*30)

def clean_dataset(name, df):
    # Change from % to proportion
    percent_columns = [col for col in df.columns if df[col].astype(str).str.contains('%').any()]
    # Remove the '%' and convert to proportion (divide by 100)
    for col in percent_columns:
        df[col] = df[col].str.rstrip('%').astype(float) / 100
    # Remove columns with only one unique value
    df = df.loc[:, df.nunique(axis=0)>1]
    #Print the total number of columns
    print(f'There are {df.shape[1]} features in the {name} after no value column drop.')

    return df

# To transpose columns in suppliers
def col_transpose(df, kw):
    print('-'*30, 'end of cleaning', '-'*30)
    return df.set_index(kw).transpose()

# Data preprocessing steps: var, corr matrix
def preprocess_data(df, name):
    # Drop columns with low variances
    threshold=0.01
    numeric_columns = df.select_dtypes(include=['number'])
    
    # Calculate variance for each column
    variances = numeric_columns.var()
    
    # Select columns with variance > threshold
    selected_columns = variances[variances >= threshold].index
    
    # Create a new DataFrame with selected columns
    reduced_df = df[selected_columns]

    #Print the total number of columns
    print(f'There are {reduced_df.shape[1]} features in the {name} after variance drop.')

    # Drop columns with high correlation
    # Calculate the correlation matrix 
    corr_matrix = reduced_df.corr(numeric_only=True).abs()
    
    # Create a mask for the upper triangle of the correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation greater than the threshold=0.8
    highly_correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
    
    # Drop the highly correlated columns
    reduced_corr_df = reduced_df.drop(columns=highly_correlated_features)

    print(f'There are {reduced_corr_df.shape[1]} features in the {name} after correlation matrix reduction.')

   # Create a copy of the original DataFrame
    scaled_df = reduced_corr_df.copy()

    # Select only numeric columns
    numeric_cols = reduced_corr_df.select_dtypes(include=['number']).columns
    numeric_data = reduced_corr_df[numeric_cols]

    # Initialize MinMaxScaler to scale features in the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit the scaler to the numeric columns and transform the data
    scaled_data = scaler.fit_transform(numeric_data)

    # Replace the numeric columns in the copy with the scaled values
    scaled_df[numeric_cols] = scaled_data
    
    return scaled_df

# Check tasks that are not present in cost_df
def filter_not_in_costs(df, cost):
    # Filter tasks to include only rows with Task IDs present in cost_df
    filtered_df = df[df.index.isin(cost.index)]
    return filtered_df

# USe iqr to drop suppliers with high errors of Cost sorted by median
def iqr_drop_suppliers(cost_df, suppliers_df):
    # Group data by 'Supplier ID' and calculate the median and IQR
    supplier_medians = cost_df.groupby('Supplier ID')['Cost'].median()
    print(supplier_medians.shape)

    q1 = supplier_medians.quantile(0.25)  
    q3 = supplier_medians.quantile(0.75)  
    iqr = q3 - q1                

    # Calculate the upper bound for outliers
    upper_bound = q3 + 1.5 * iqr

    # Drop suppliers whose median cost exceeds the upper bound
    valid_suppliers = supplier_medians[supplier_medians < upper_bound].index

    # Check valid suppliers
    print("\nValid Suppliers:", valid_suppliers.tolist())
    print(f"\nNumber of Valid Suppliers: {len(valid_suppliers)}")

    # Filter cost_df to include only valid suppliers
    filtered_cost_df = cost_df[cost_df['Supplier ID'].isin(valid_suppliers)]

    # Check the shape and unique supplier IDs in the filtered cost DataFrame
    print("\nFiltered Cost DataFrame Shape:", filtered_cost_df.shape)
    print("\nUnique Supplier IDs in Filtered Cost DataFrame:", filtered_cost_df['Supplier ID'].nunique())

    # Identify suppliers to exclude
    excluded_suppliers = supplier_medians[supplier_medians > upper_bound].index

    # Check excluded suppliers
    print("\nSuppliers to Exclude:", excluded_suppliers.tolist())
    print(f"\nNumber of Suppliers to Exclude: {len(excluded_suppliers)}")

    filtered_suppliers_df = suppliers_df[~suppliers_df.index.isin(excluded_suppliers)]
    print("\nFiltered Suppliers DataFrame Shape:", filtered_suppliers_df.shape)

    print("\nUnique Supplier IDs in cost_df:", filtered_cost_df['Supplier ID'].unique())
    print("\nValid Supplier IDs in suppliers_df:", filtered_suppliers_df.index.unique())
    return filtered_cost_df, filtered_suppliers_df

# Use iqr to exclude suppliers by rmse
def high_error_supp_drop(cost_df, suppliers_df):
    # Create a copy of cost_df for calculations
    costs_error = cost_df.copy()

    # Check if 'Task ID' is a column or the index
    if 'Task ID' not in costs_error.columns:
        if costs_error.index.name == 'Task ID':
            costs_error.reset_index(inplace=True)
        else:
            raise KeyError("'Task ID' is neither a column nor the index in the cost DataFrame.")

    # Identify the minimum cost for each task
    best_cost = costs_error.groupby('Task ID')['Cost'].min()
    costs_error.set_index('Task ID', inplace=True)
    costs_error['Min_Cost'] = best_cost

    # Calculate the error: Difference between the minimum cost and the supplier's cost
    costs_error['Error'] = costs_error['Min_Cost'] - costs_error['Cost']

    # Compute RMSE for each supplier
    rmse_per_supplier = costs_error.groupby('Supplier ID')['Error'].apply(lambda x: np.sqrt((x ** 2).mean()))
    print(rmse_per_supplier)

    # Calculate the 75th percentile threshold for RMSE
    threshold = rmse_per_supplier.quantile(0.75)

    # Filter suppliers: Below or equal to the threshold are kept
    filtered_suppliers = rmse_per_supplier[rmse_per_supplier <= threshold]

    # Suppliers above the threshold are excluded
    excluded_suppliers = rmse_per_supplier[rmse_per_supplier > threshold]

    # Output results
    print(f"75th Percentile Threshold: {threshold:.4f}")
    print("\nSuppliers kept (below or equal to threshold):")
    print(filtered_suppliers)

    print("\nSuppliers removed (above threshold):")
    print(excluded_suppliers)

    # Filter cost_df to include only valid suppliers
    final_cost_df = cost_df[cost_df['Supplier ID'].isin(filtered_suppliers.index)]

    # Verify the filtered cost DataFrame
    print("Filtered Cost DataFrame Shape:", final_cost_df.shape)
    print("Unique Supplier IDs in Filtered Cost DataFrame:", final_cost_df['Supplier ID'].nunique())

    # Filter suppliers_df to include only valid suppliers
    final_supplier_df = suppliers_df[suppliers_df.index.isin(filtered_suppliers.index)]

    # Verify the filtered supplier DataFrame
    print("Filtered Supplier DataFrame Shape:", final_supplier_df.shape)
    print("Unique Supplier IDs in Filtered Supplier DataFrame:", final_supplier_df.index.nunique())

    return final_cost_df, final_supplier_df


