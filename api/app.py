from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app) # Enable CORS for frontend

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/bankruptcy_model.pkl"))
model_data = None

def load_model():
    global model_data
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
    return model_data

load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        # Try loading again in case it was just trained
        if not load_model():
            return jsonify({'error': 'Model not trained or missing.'}), 500
        
    try:
        data = request.json
        features = model_data['features']
        
        # Ensure all required features are present
        input_vector = []
        for f in features:
            if f not in data:
                return jsonify({'error': f'Missing feature: {f}'}), 400
            input_vector.append(float(data[f]))
            
        input_array = np.array(input_vector).reshape(1, -1)
        scaled_input = model_data['scaler'].transform(input_array)
        
        prob = model_data['model'].predict_proba(scaled_input)[0, 1]
        pred = model_data['model'].predict(scaled_input)[0]
        
        risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
        
        return jsonify({
            'prediction': risk_level,
            'probability': round(float(prob), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/features', methods=['GET'])
def get_features():
    if model_data is None:
        if not load_model():
            return jsonify({'error': 'Model not trained or missing.'}), 500
    return jsonify({'features': model_data['features']})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
