# This module is for handling outliers.

import pandas as pd

def handle_outliers(df):
    # Here you can add logic to handle outliers, for example, using IQR method
    # For demonstration, let's assume we are capping values to the 1st and 99th percentiles
    numerical_columns = df.select_dtypes(include=[float, int]).columns
    for column in numerical_columns:
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df
