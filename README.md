# 🫀 MediRisk AI

### AI-Powered Heart Disease Risk Prediction System

MediRisk AI is a machine learning-based web application designed to estimate potential heart disease risk using clinical health parameters.

The system analyzes patient information using trained Machine Learning models and provides risk predictions, probability scores, risk classification, important contributing factors, health insights, and an AI-powered health assistant.

> ⚠️ **Medical Disclaimer:** This project is developed for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 🚀 Features

- 🫀 Heart Disease Risk Prediction
- 📊 Risk Probability Analysis
- 🎯 Low / Moderate / High Risk Classification
- 💯 Health Score
- 📈 Top Risk Factors using Feature Importance
- 🔬 What-If Risk Simulator
- 📋 Assessment Summary
- 📈 Risk History
- 💡 Personalized Recommendations
- 🤖 AI Health Assistant
- 💬 Interactive Chat Interface
- 📄 Downloadable PDF Risk Report
- 🔐 Environment Variable-Based API Key Protection
- 📱 Responsive Web Interface
- ✨ Modern UI with Animations
- ⚡ Fast Flask Backend

---

## 🧠 Machine Learning

The project evaluates multiple Machine Learning algorithms and compares their performance to select the most suitable model.

### Models Tested

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### 🏆 Best Performing Model

**Random Forest Classifier**

| Metric | Score |
|---|---:|
| Accuracy | **86.67%** |
| Precision | **88.46%** |
| Recall | **82.14%** |
| F1 Score | **85.19%** |
| ROC-AUC | **94.20%** |

The Random Forest model is used as the final prediction model.

---

## 📊 Dataset

The project uses the **UCI Heart Disease Dataset (Cleveland)**.

### Dataset Information

- **Original records:** 303
- **Records after removing missing values:** 297
- **Input features:** 13
- **Target:** Heart Disease
- **Classification:** Binary

### Target Classes

| Value | Meaning |
|---|---|
| `0` | No Heart Disease |
| `1` | Potential Heart Disease |

---

## 📥 Input Features

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | Patient sex |
| CP | Chest pain type |
| Trestbps | Resting blood pressure |
| Chol | Serum cholesterol |
| FBS | Fasting blood sugar |
| Restecg | Resting ECG result |
| Thalach | Maximum heart rate achieved |
| Exang | Exercise-induced angina |
| Oldpeak | ST depression |
| Slope | ST segment slope |
| CA | Number of major vessels |
| Thal | Thalassemia |

---

## 🛠️ Technologies Used

### Frontend

- HTML5
- CSS3
- JavaScript
- Font Awesome

### Backend

- Python
- Flask

### Machine Learning

- Scikit-learn
- XGBoost
- NumPy
- Pandas
- Joblib

### Artificial Intelligence

- OpenAI API

### Reporting

- ReportLab

---

## 🎨 Professional UI

MediRisk AI includes a modern, responsive, portfolio-focused healthcare interface featuring:

- Premium dark medical/AI visual design
- Responsive patient assessment form
- BMI auto-calculation
- Risk probability visualization
- Health score presentation
- Feature importance and explainable insights
- Personalized recommendations
- What-If risk simulation
- Risk history and trend visualization
- Assessment summary
- AI Health Assistant
- Interactive chat interface
- Downloadable PDF assessment report
- Mobile-friendly responsive layouts

---

## 🤖 AI Health Assistant

The integrated AI Health Assistant helps users understand their assessment results in simple language.

It can provide educational explanations about:

- Risk prediction
- Important risk factors
- Health parameters
- Cholesterol
- Assessment results
- Practical next steps
- Questions to discuss with a healthcare professional

The AI assistant does not provide medical diagnosis or prescribe medication.

---

## 📈 Explainable AI

MediRisk AI is designed to provide more than a simple prediction.

The application displays important model factors that contribute to the prediction, helping users understand which clinical parameters have greater influence on the model's result.

This demonstrates the practical application of **Explainable AI (XAI)** in a healthcare-oriented Machine Learning project.

---

## 🔬 What-If Simulator

The What-If Simulator allows users to modify selected health parameters and observe how the model's prediction changes.

```text
Current Assessment
        ↓
Modify Health Parameter
        ↓
Run Simulation
        ↓
Model Prediction
        ↓
Compare Results
