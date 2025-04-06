import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

# Load models and scaler
rf_model = joblib.load('rf_fraud.pkl')
ae_model = tf.keras.models.load_model('ae_fraud.keras')  # Updated to .keras
scaler = joblib.load('scaler_fraud.pkl')

# Feature names
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                 'V28', 'Amount']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get input from form
            inputs = [float(request.form.get(feat, 0)) for feat in feature_names]
            input_array = np.array([inputs])
            input_scaled = scaler.transform(input_array)

            # Random Forest prediction
            rf_prob = rf_model.predict_proba(input_scaled)[0][1] * 100  # % fraud
            rf_pred = rf_model.predict(input_scaled)[0]
            rf_importance = rf_model.feature_importances_
            top_features = sorted(zip(feature_names, rf_importance), key=lambda x: x[1], reverse=True)[:3]

            # Autoencoder prediction
            reconstructed = ae_model.predict(input_scaled, verbose=0)
            recon_error = np.mean((input_scaled - reconstructed) ** 2)
            threshold = 0.05  # Adjust to your Day 14 threshold (e.g., from rerun)
            ae_prob = min(recon_error / threshold * 100, 100)  # Cap at 100%

            return render_template('index.html', 
                                 rf_prob=f"{rf_prob:.2f}", 
                                 rf_pred=rf_pred,
                                 ae_prob=f"{ae_prob:.2f}",
                                 top_features=top_features,
                                 inputs=inputs)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)