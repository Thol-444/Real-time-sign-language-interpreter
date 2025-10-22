# train_model.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load collected data
data_file = 'data/sign_data.csv'
if not os.path.exists(data_file):
    raise FileNotFoundError("❌ Data file not found! Run collect_data.py first to generate sign_data.csv.")

df = pd.read_csv(data_file)

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/sign_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("💾 Saved model/sign_model.pkl and model/scaler.pkl")
