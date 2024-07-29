import pandas as pd
import numpy as np

def validate_data(df):
    validation_errors = []

    validation_rules = {
        'Gender': {'type': 'categorical', 'values': ['Male', 'Female']},
        'Age': {'type': 'continuous', 'range': (0, 80)},
        'Region': {'type': 'categorical', 'values': ['North-West', 'North-East', 'Center', 'South & Islands']},
        'Occupation': {'type': 'categorical', 'values': ['Tier one', 'Requiring extra underwriting']},
        'Physical Examination': {'type': 'categorical', 'values': ['Exempted', 'Required']},
        'Company': {'type': 'categorical', 'values': ['Company A', 'Company B', 'Company C']},
        'Time from Start': {'type': 'continuous', 'range': (0, None)},
        'Time to Expiry': {'type': 'continuous', 'range': (0, None)},
        'Contract Size': {'type': 'continuous', 'range': (0, None)},
        'Premium Type': {'type': 'categorical', 'values': ['Single', 'Non-single']},
        'Product Type': {'type': 'categorical', 'values': ['Traditional', 'Unit-linked', 'Interest-adjustable', 'Investment-linked', 'Conventional']},
        'Premium Payment Frequency': {'type': 'categorical', 'values': ['Monthly', 'Quarterly', 'Semi-annual', 'Annual']},
        'Policy Duration': {'type': 'continuous', 'range': (0, None)},
        'Disposable Income Growth Rate': {'type': 'continuous', 'range': (None, None)},
        'Inflation Rate': {'type': 'continuous', 'range': (None, None)},
        'Eurostoxx Index Growth Rate': {'type': 'continuous', 'range': (None, None)},
        'Risk-Free Rate': {'type': 'continuous', 'range': (None, None)},
        'Distribution Channel': {'type': 'categorical', 'values': ['TA', 'BK', 'DM', 'Others']},
        'Number of Customer Service Interactions': {'type': 'continuous', 'range': (0, None)},
        'Number of Complaints': {'type': 'continuous', 'range': (0, None)},
        'Satisfaction Score': {'type': 'continuous', 'range': (1, 10)},
        'Number of Late Payments': {'type': 'continuous', 'range': (0, None)},
        'Payment Method': {'type': 'categorical', 'values': ['Insurer B&C', 'P&C', 'Other']},
        'Payment Frequency': {'type': 'categorical', 'values': ['Monthly', 'Quarterly', 'Semi-annual', 'Annual']},
        'Payment Consistency': {'type': 'categorical', 'values': ['Consistent', 'Inconsistent']},
        'Starting Date': {'type': 'date'},
        'Expiry Date': {'type': 'date'},
        'Lapse Date': {'type': 'date'},
        'Face Amounts': {'type': 'continuous', 'range': (0, None)},
        'Number of Non-life Policies': {'type': 'continuous', 'range': (0, None)},
        'Number of Claims Filed': {'type': 'continuous', 'range': (0, None)},
        'Total Claim Amount': {'type': 'continuous', 'range': (0, None)},
        'Claim Frequency': {'type': 'continuous', 'range': (0, None)},
        'Engagement Score': {'type': 'continuous', 'range': (1, 10)},
        'Loyalty Program Participation': {'type': 'categorical', 'values': ['Yes', 'No']},
        'Riders Attached to Policy': {'type': 'categorical'},
        'Policy Benefits Utilization': {'type': 'categorical', 'values': ['Utilized', 'Not Utilized']},
        'Credit Score': {'type': 'continuous', 'range': (300, 850)},
        'Debt-to-Income Ratio': {'type': 'continuous', 'range': (0, None)},
        'Preferred Communication Channel': {'type': 'categorical', 'values': ['Email', 'Phone', 'SMS', 'In-person']},
        'Preferred Payment Method': {'type': 'categorical', 'values': ['Credit Card', 'Bank Transfer', 'Direct Debit']}
    }

    for column, rules in validation_rules.items():
        if column not in df.columns:
            validation_errors.append(f"Missing column: {column}")
            continue

        # Check for missing values
        missing_values = df[df[column].isnull()]
        for index, row in missing_values.iterrows():
            validation_errors.append(f"Missing value in column '{column}' at row {index}")

        if rules['type'] == 'categorical':
            if 'values' in rules:
                invalid_values = df[~df[column].isin(rules['values'])]
                for index, row in invalid_values.iterrows():
                    validation_errors.append(f"Invalid value '{row[column]}' in column '{column}' at row {index}")

        if rules['type'] == 'continuous':
            if 'range' in rules:
                if rules['range'][0] is not None:
                    invalid_values = df[df[column] < rules['range'][0]]
                    for index, row in invalid_values.iterrows():
                        validation_errors.append(f"Value '{row[column]}' in column '{column}' at row {index} is less than minimum allowed value {rules['range'][0]}")
                if rules['range'][1] is not None:
                    invalid_values = df[df[column] > rules['range'][1]]
                    for index, row in invalid_values.iterrows():
                        validation_errors.append(f"Value '{row[column]}' in column '{column}' at row {index} is greater than maximum allowed value {rules['range'][1]}")

        if rules['type'] == 'date':
            try:
                pd.to_datetime(df[column])
            except ValueError:
                invalid_values = df[pd.to_datetime(df[column], errors='coerce').isna()]
                for index, row in invalid_values.iterrows():
                    validation_errors.append(f"Invalid date format '{row[column]}' in column '{column}' at row {index}")

    return validation_errors
