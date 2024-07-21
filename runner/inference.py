import pandas as pd
import logging
import joblib
import os
from src.data_cleaning import clean_data
from src.feature_generation import generate_features
from src.handling_missing_values import handle_missing_values
from src.handling_outliers import handle_outliers

# Set up simple logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.debug(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    logger.debug("Starting data preprocessing...")
    df = clean_data(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    categorical_columns = ['Gender', 'Region', 'Occupation', 'Physical Examination', 'Company',
       'Premium Type', 'Product Type', 'Premium Payment Frequency',
       'Distribution Channel', 'Payment Method', 'Payment Frequency','Payment Consistency',
       'Loyalty Program Participation', 'Riders Attached to Policy',
       'Policy Benefits Utilization', 'Preferred Communication Channel',
       'Preferred Payment Method']
    df = generate_features(df, categorical_columns)
    logger.debug("Data preprocessing completed.")
    return df

def load_models(model_dir='models'):
    logger.debug("Loading models...")
    models = {}
    for model_name in ['xgboost_model']:  # Considering best model out of 'logistic_regression', 'random_forest', 'xgboost_model'
        model_path = os.path.join(model_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
            logger.debug(f"Loaded {model_name} model from {model_path}")
        else:
            logger.error(f"Model file {model_path} does not exist.")
    return models

def batch_inference(df, models):
    logger.debug("Starting batch inference...")
    X = df.drop(columns=['Label'], errors='ignore')  # Ensure 'Label' column is not included in the prediction data
    inference_dict = {}
    for model_name, model in models.items():
        logger.debug(f"Predicting with {model_name} model...")
        inference_dict['predictions'] = model.predict(X)
        inference_dict['probabilities'] = model.predict_proba(X)[:, 1]  # Get probability of the positive class
    return inference_dict

def save_predictions(input_data_path, inference_dict, output_file='data/inference.csv'):
    logger.debug(f"Saving predictions to {output_file}")
    logger.debug(f"inference_dict: {inference_dict}")
    # Saving to original data path as such It's more understanable to general user.
    df = load_data(input_data_path)
    df['churn_label'] = inference_dict['predictions']
    df['churn_score'] = inference_dict['probabilities']*100
    df['churn_score'] = df['churn_score'].round(0).astype(int)
    df.to_csv(output_file, index=False)
    logger.debug("Predictions saved successfully.")

if __name__ == "__main__":
    data_path = 'data/raw.csv'
    output_file = 'data/processed.csv'
    
    logger.debug("Starting inference process...")
    df = load_data(data_path)
    df = preprocess_data(df)
    models = load_models()
    inference_dict = batch_inference(df, models)
    save_predictions(df, inference_dict, output_file)
    logger.debug("Inference process completed.")
