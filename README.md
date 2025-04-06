🔐 Fraud Detection System

A machine learning & deep learning-powered web app built using Flask to detect fraudulent credit card transactions in real time. This project uses advanced models to identify anomalies based on historical transaction patterns.

🧠 Models Used

Random Forest Classifier – A supervised model trained on labeled data.

Autoencoder Neural Network – An unsupervised deep learning model for anomaly detection.

Both models work together to provide high-accuracy predictions and highlight suspicious activities.

🚀 Key Features

✅ Predicts if a transaction is fraudulent or not📊 Displays anomaly score and prediction confidence📉 Scaled inputs to enhance model performance🌐 User-friendly web interface built with Flask📌 Highlights top influential features in the prediction

🛠️ Tech Stack

Component

Technology

Language

Python

ML Libraries

Scikit-learn, TensorFlow/Keras

Web Framework

Flask

Frontend

HTML, CSS

Dataset

Kaggle Credit Card Fraud Dataset

📂 Project Structure

fraud-detection-system/
├── app.py                 # Main Flask app
├── rf_model.pkl           # Random Forest model
├── autoencoder_model.h5   # Autoencoder neural network
├── scaler.pkl             # Preprocessing scaler
├── templates/
│   └── index.html         # Web interface template
├── static/
│   └── style.css          # Styling (CSS)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

💻 Installation & Run

Clone the repository

git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

Run the application

python app.py

Then open your browser and navigate to:🔗 http://127.0.0.1:5000/
