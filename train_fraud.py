import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import joblib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load dataset
print("Loading data...")
data = pd.read_csv('creditcard.csv')
data.columns = data.columns.str.strip()
print("Column names:", data.columns.tolist())

if 'Class' not in data.columns:
    raise ValueError("'Class' column not found in dataset.")

# Split features and labels
X = data.drop('Class', axis=1).values
y = data['Class'].values

# Split dataset
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balance data with SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"SMOTE done. Balanced shape: {X_train_balanced.shape}")

# 1. Random Forest Classifier
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_balanced, y_train_balanced)
rf_pred = rf.predict(X_test_scaled)
rf_prob = rf.predict_proba(X_test_scaled)[:, 1]

print("\n🔍 Random Forest Results:")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

# 2. Autoencoder
print("Building Autoencoder...")
input_dim = X_train_scaled.shape[1]
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')

# Train on non-fraudulent data only
X_train_legit = X_train_scaled[y_train == 0]
print(f"Training Autoencoder on {X_train_legit.shape[0]} legit samples...")
autoencoder.fit(X_train_legit, X_train_legit, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Detect anomalies
print("Evaluating Autoencoder...")
X_test_reconstructed = autoencoder.predict(X_test_scaled)
reconstruction_error = np.mean((X_test_scaled - X_test_reconstructed) ** 2, axis=1)
threshold = np.percentile(reconstruction_error[y_test == 0], 95)
ae_pred = (reconstruction_error > threshold).astype(int)

print("\n🧠 Autoencoder Results:")
print(classification_report(y_test, ae_pred))
print("ROC-AUC (Reconstruction Error):", roc_auc_score(y_test, reconstruction_error))

# Save models and scaler
print("Saving models...")
joblib.dump(rf, 'rf_fraud.pkl')
autoencoder.save('ae_fraud.keras')  # ✅ modern format for compatibility
joblib.dump(scaler, 'scaler_fraud.pkl')
print("\n✅ Models saved: 'rf_fraud.pkl', 'ae_fraud.h5', 'scaler_fraud.pkl'")