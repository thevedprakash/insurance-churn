import os
import joblib
import logging
import pandas as pd
from src.data_cleaning import clean_data
from src.feature_generation import generate_features
from src.handling_missing_values import handle_missing_values
from src.handling_outliers import handle_outliers
from sklearn.model_selection import train_test_split
from src.models_training import logistic_regression, random_forest, xgboost_model

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

def train_and_save_models(df):
    logger.debug("Starting model training...")
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    logger.debug("Training logistic regression model...")
    logistic_regression_model = logistic_regression.train(X_train, y_train)
    logger.debug("Training random forest model...")
    random_forest_model = random_forest.train(X_train, y_train)
    logger.debug("Training XGBoost model...")
    xgboost_model_instance = xgboost_model.train(X_train, y_train)
    
    models = {
        'logistic_regression': logistic_regression_model,
        'random_forest': random_forest_model,
        'xgboost_model': xgboost_model_instance
    }
    
    # Dictionary to map model names to their respective modules
    eval_functions = {
        'logistic_regression': logistic_regression.evaluate,
        'random_forest': random_forest.evaluate,
        'xgboost_model': xgboost_model.evaluate
    }
    
    # Evaluate models
    for model_name, model in models.items():
        logger.debug(f"Evaluating {model_name} model...")
        eval_results = eval_functions[model_name](model, X_test, y_test)
        logger.info(f"Evaluation results for {model_name}:")
        for metric, value in eval_results.items():
            if metric != 'classification_report':
                logger.info(f"{metric}: {value}")
            else:
                logger.info(value)
    
    # Save models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    for model_name, model in models.items():
        model_path = f'models/{model_name}.pkl'
        joblib.dump(model, model_path)
        logger.debug(f"Saved {model_name} model to {model_path}")
    
    logger.debug("Model training and saving completed.")

if __name__ == "__main__":
    data_path = 'data/raw.csv'
    logger.debug("Starting main process...")
    df = load_data(data_path)
    df = preprocess_data(df)
    train_and_save_models(df)
    logger.debug("Main process completed.")
