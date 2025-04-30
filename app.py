from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# Load scaler and models
try:
    scaler = joblib.load('data/scaler.pkl')
except FileNotFoundError as e:
    raise FileNotFoundError(f"Scaler file 'data/scaler.pkl' not found: {str(e)}. Ensure the file exists in the 'data/' directory.")
except Exception as e:
    raise Exception(f"Error loading scaler: {str(e)}")

models = {}
model_files = {
    'bayesian': 'models/bayesian_model.pkl',
    'naive_bayes': 'models/naive_bayes_model.pkl',
    'random_forest': 'models/random_forest_model.pkl'
}

for model_name, model_path in model_files.items():
    if os.path.exists(model_path):
        try:
            models[model_name] = joblib.load(model_path)
        except Exception as e:
            raise Exception(f"Error loading model '{model_name}' from '{model_path}': {str(e)}")
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure all model files exist in the 'models/' directory.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/comparison')
def comparison():
    accuracies = {
        'Bayesian': 0.78,
        'Naive Bayes': 0.75,
        'Random Forest': 0.85
    }
    return render_template('comparison.html', accuracies=accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['dpf']),
            float(data['age'])
        ]
        selected_models = json.loads(data.get('models', '[]'))
        if not selected_models:
            selected_models = list(models.keys())

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        predictions = {}
        for model_name in selected_models:
            if model_name in models:
                pred = models[model_name].predict(features_scaled)[0]
                predictions[model_name] = 'Diabetic' if pred == 1 else 'Non-Diabetic'
            else:
                return jsonify({'error': f'Model {model_name} not found'}), 400
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)