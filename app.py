from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
from runner.inference import load_data, preprocess_data, load_models, batch_inference, save_predictions

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Ensure the upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Run inference
        df = load_data(file_path)
        df = preprocess_data(df)
        models = load_models()
        inference_dict = batch_inference(df, models)
        
        output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'predictions.csv')
        save_predictions(file_path, inference_dict, output_file_path)

        
        # loading original Data again to display with churn prediction and Churn probabilities.
        # Reload the original data with predictions for display
        display_df = load_data(file_path)
        display_df['churn_label'] = inference_dict['predictions']
        display_df['churn_score'] = inference_dict['probabilities'] * 100
        display_df['churn_score'] = display_df['churn_score'].round(0).astype(int)

        # Convert DataFrame to dictionary for rendering in template
        df_dict = display_df.to_dict(orient='records')

        return render_template('index.html', filename='predictions.csv', df=df_dict)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
