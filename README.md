# 🫀 MediRisk AI

### AI-Powered Heart Disease Risk Prediction System

MediRisk AI is a machine learning-based web application designed to estimate a patient's potential heart disease risk using clinical health parameters.

The system analyzes patient information using a trained Machine Learning model and provides a risk prediction along with probability scores, important risk factors, and an AI-powered health assistant.

> ⚠️ This project is for educational and demonstration purposes only. It is not a substitute for professional medical diagnosis or treatment.

---

## 🚀 Features

- 🫀 Heart Disease Risk Prediction
- 📊 Risk Probability Analysis
- 🎯 Low / Moderate / High Risk Classification
- 📈 Top Risk Factors using Feature Importance
- 🤖 AI Health Assistant
- 💬 Interactive Chat Interface
- 📄 Downloadable Medical Risk Report
- 🔐 Environment Variable Based API Key Protection
- 📱 Responsive Web Interface
- ✨ Modern UI with Animations
- ⚡ Fast Flask Backend

---

## 🧠 Machine Learning

The project uses multiple Machine Learning algorithms and compares their performance.

### Models Tested

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Best Performing Model

**Random Forest**

| Metric | Score |
|---|---:|
| Accuracy | 86.67% |
| Precision | 88.46% |
| Recall | 82.14% |
| F1 Score | 85.19% |
| ROC-AUC | 94.20% |

The Random Forest model is used as the final prediction model.

---

## 📊 Dataset

The project uses the **UCI Heart Disease Dataset (Cleveland)**.

### Dataset Information

- Total original records: **303**
- Records after removing missing values: **297**
- Features: **13**
- Target: **Heart Disease**
- Classes:
  - `0` → No Heart Disease
  - `1` → Potential Heart Disease

### Input Features

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | Gender |
| CP | Chest pain type |
| Trestbps | Resting blood pressure |
| Chol | Cholesterol |
| FBS | Fasting blood sugar |
| Restecg | Resting ECG |
| Thalach | Maximum heart rate |
| Exang | Exercise induced angina |
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

### AI
- OpenAI API

### Reporting
- ReportLab

---

## 📁 Project Structure

```text
MediRisk_AI/
│
├── dataset/
│   └── heart.csv
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── predict.py
├── best_model.pkl
├── scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md

## Professional UI

MediRisk AI uses a responsive, portfolio-focused interface with:
- Premium dark medical/AI visual design
- Responsive patient assessment form
- BMI auto-calculation
- Risk probability and health-score presentation
- Feature-importance explanation
- AI health assistant
- Downloadable PDF assessment report
- Mobile-friendly layouts

> This project is for educational/demo purposes and is not a medical diagnostic system.
