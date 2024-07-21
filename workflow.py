# This module is for running taining and inference.

import sys
import logging
from runner.train_model import train_and_save_models
from runner.inference import load_data, preprocess_data, load_models, batch_inference, save_predictions

# Set up simple logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main(task):
    if task == 'train':
        logger.debug("Starting model training...")
        data_path = 'data/raw.csv'
        df = load_data(data_path)
        logger.debug("Load Data completed.")
        df = preprocess_data(df)
        logger.debug("Processing data completed.")
        train_and_save_models(df)
        logger.debug("Model training completed.")
    elif task == 'inference':
        logger.debug("Starting batch inference...")
        data_path = 'data/test.csv'
        output_file = 'data/inference1.csv'
        df = load_data(data_path)
        logger.debug("Load Data completed.")
        df = preprocess_data(df)
        logger.debug("Processing data completed.")
        models = load_models()
        logger.debug("Model Loaded for Inference completed.")
        predictions = batch_inference(df, models)
        logger.debug("Model Generated Inference completed.")
        save_predictions(data_path, predictions, output_file)
        logger.debug("Batch inference completed.")
    else:
        logger.error("Invalid task. Please specify 'train' or 'inference'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python run.py <task>")
        logger.error("task: 'train' or 'inference'")
        sys.exit(1)
    task = sys.argv[1]
    main(task)
