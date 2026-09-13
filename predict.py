import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

print("\n===================================")
print("      MEDIRISK AI")
print("   Heart Disease Risk Prediction")
print("===================================\n")

# Patient information
age = float(input("Age: "))
sex = float(input("Sex (1 = Male, 0 = Female): "))
cp = float(input("Chest Pain Type (1-4): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar (1 = Yes, 0 = No): "))
restecg = float(input("Resting ECG (0-2): "))
thalach = float(input("Maximum Heart Rate: "))
exang = float(input("Exercise Induced Angina (1 = Yes, 0 = No): "))
oldpeak = float(input("ST Depression (oldpeak): "))
slope = float(input("Slope (1-3): "))
ca = float(input("Number of Major Vessels (0-3): "))
thal = float(input("Thalassemia (3, 6, 7): "))

# Create input array
patient_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=[
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
])

# Scale input
patient_scaled = scaler.transform(patient_data)

# Prediction
prediction = model.predict(patient_scaled)[0]

# Probability
probabilities = model.predict_proba(patient_scaled)[0]

disease_probability = probabilities[1] * 100
no_disease_probability = probabilities[0] * 100

print("\n===================================")
print("         RISK ASSESSMENT")
print("===================================\n")

if prediction == 1:
    print("Prediction          : HEART DISEASE RISK")
else:
    print("Prediction          : NO HEART DISEASE RISK")

print(f"Disease Probability : {disease_probability:.2f}%")
print(f"No Disease          : {no_disease_probability:.2f}%")

print("\n===================================")
print("⚠️ This is an ML-based risk prediction")
print("   and NOT a medical diagnosis.")
print("===================================\n")