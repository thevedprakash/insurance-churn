from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
from runner.inference import load_data, preprocess_data, load_models, batch_inference, save_predictions
from utils.utils import validate_data

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this to a secure random key in production

CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure the upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.verify_password(password):
            login_user(user)
            return redirect(url_for('upload'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('upload'))
        
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('upload'))
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Load the data
            df = load_data(file_path)

            # Validate the data
            validation_errors = validate_data(df)
            if validation_errors:
                return render_template('error.html', validation_errors=validation_errors, filename=file.filename)

            return process_file(file.filename)
    return render_template('index.html')

@app.route('/process/<filename>')
@login_required
def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_data(file_path)
    df = preprocess_data(df)
    models = load_models()
    inference_dict = batch_inference(df, models)
    
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'predictions.csv')
    save_predictions(file_path, inference_dict, output_file_path)

    # Reload the original data with predictions for display
    display_df = load_data(file_path)
    display_df['churn_label'] = inference_dict['predictions']
    display_df['churn_score'] = inference_dict['probabilities'] * 100
    display_df['churn_score'] = display_df['churn_score'].round(0).astype(int)

    # Convert DataFrame to dictionary for rendering in template
    df_dict = display_df.to_dict(orient='records')

    return render_template('index.html', filename='predictions.csv', df=df_dict)

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
