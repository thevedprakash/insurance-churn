# This module is for data cleaning operations.

import pandas as pd

# Function to extract date features
def extract_date_features(df):
    df['Starting Date'] = pd.to_datetime(df['Starting Date'])
    df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])
    df['Lapse Date'] = pd.to_datetime(df['Lapse Date'])

    df['Start Year'] = df['Starting Date'].dt.year
    df['Start Month'] = df['Starting Date'].dt.month
    df['Expiry Year'] = df['Expiry Date'].dt.year
    df['Expiry Month'] = df['Expiry Date'].dt.month
    df['Lapse Year'] = df['Lapse Date'].dt.year
    df['Lapse Month'] = df['Lapse Date'].dt.month

    df['Policy Duration Days'] = (df['Expiry Date'] - df['Starting Date']).dt.days
    df['Lapse Duration Days'] = (df['Lapse Date'] - df['Starting Date']).dt.days

    return df.drop(columns=['Starting Date', 'Expiry Date', 'Lapse Date'])

def clean_data(df):
    # Additional cleaning steps can be added here
    df = extract_date_features(df)
    return df



