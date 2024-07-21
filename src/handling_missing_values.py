# This module is for handling missing values.

from sklearn.impute import SimpleImputer
import pandas as pd

def handle_missing_values(df):
    # Define the numerical columns
    numerical_columns = df.select_dtypes(include=[float, int]).columns
    
    # Create an imputer
    imputer = SimpleImputer(strategy='median')
    
    # Fit and transform the data
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    
    return df
