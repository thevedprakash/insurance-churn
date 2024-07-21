# This module is for feature generation.

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import json

# Transformer to handle categorical mapping
class CategoricalMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns, strategy='default', save_to_file=False, filename=None):
        self.categorical_columns = categorical_columns
        self.strategy = strategy
        self.save_to_file = save_to_file
        self.filename = filename
        self.mappings_dict = {}

    def create_categorical_mappings(self, dataframe, strategy):
        mappings_dict = {}
        for column in self.categorical_columns:
            if strategy == 'default':
                unique_values = dataframe[column].unique()
                value_to_label = {value: int(idx) for idx, value in enumerate(unique_values)}
            elif strategy == 'count':
                value_counts = dataframe[column].value_counts()
                value_to_label = {value: int(count) for value, count in value_counts.items()}
            elif strategy == 'percentage':
                value_counts = dataframe[column].value_counts(normalize=True) * 100
                value_to_label = {value: float(percentage) for value, percentage in value_counts.items()}
            elif strategy == 'labelencoder':
                le = LabelEncoder()
                le.fit(dataframe[column])
                value_to_label = {value: int(label) for value, label in zip(le.classes_, le.transform(le.classes_))}
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            mappings_dict[column] = value_to_label
        return mappings_dict

    def transform_categorical_data(self, dataframe, mappings_dict):
        transformed_df = dataframe.copy()
        for column, mapping in mappings_dict.items():
            transformed_df[column] = transformed_df[column].map(mapping)
        return transformed_df

    def save_mappings_to_file(self, mappings_dict, filename):
        with open(filename, 'a') as file:
            json.dump(mappings_dict, file)
            file.write('\n')

    def fit(self, X, y=None):
        self.mappings_dict = self.create_categorical_mappings(X, self.strategy)
        if self.save_to_file and self.filename:
            self.save_mappings_to_file(self.mappings_dict, self.filename)
        return self

    def transform(self, X):
        return self.transform_categorical_data(X, self.mappings_dict)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def generate_features(df, categorical_columns):
    # Transform categorical columns
    categorical_transformer = CategoricalMappingTransformer(categorical_columns=categorical_columns, strategy='default', save_to_file=True, filename='mappings.txt')
    df = categorical_transformer.fit_transform(df)
    return df
