from flask import Flask, render_template, request, session, jsonify, send_file
import os
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Optional OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Optional ReportLab
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Flowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# APP CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "medirisk-local-secret-key"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# ============================================================
# DISPLAY MAPS
# ============================================================

DISPLAY_MAPS = {
    "smoking": {
        0: "No",
        1: "Yes"
    },

    "diabetes": {
        0: "No",
        1: "Yes"
    },

    "family_history": {
        0: "No",
        1: "Yes"
    },

    "physical_activity": {
        1: "Low",
        2: "Moderate",
        3: "High"
    },

    "stress_level": {
        1: "Low",
        2: "Moderate",
        3: "High"
    },

    "sex": {
        0: "Female",
        1: "Male"
    },

    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic"
    },

    "fbs": {
        0: "No",
        1: "Yes"
    },

    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    },

    "exang": {
        0: "No",
        1: "Yes"
    },

    "slope": {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping"
    },

    "ca": {
        0: "0 vessels",
        1: "1 vessel",
        2: "2 vessels",
        3: "3 vessels"
    },

    "thal": {
        3: "Normal",
        6: "Fixed Defect",
        7: "Reversible Defect"
    }
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def calculate_bmi(height_cm, weight_kg):
    height_cm = safe_float(height_cm)
    weight_kg = safe_float(weight_kg)

    if height_cm <= 0 or weight_kg <= 0:
        return 0.0

    height_m = height_cm / 100

    return round(
        weight_kg / (height_m ** 2),
        2
    )


def bmi_category(bmi):
    bmi = safe_float(bmi)

    if bmi <= 0:
        return "Not available"

    if bmi < 18.5:
        return "Underweight"

    if bmi < 25:
        return "Healthy range"

    if bmi < 30:
        return "Overweight"

    return "Obesity range"


# ============================================================
# EDUCATIONAL HEALTH SCORE
# ============================================================

def calculate_health_score(patient):
    score = 100

    age = safe_int(patient.get("age"))
    bmi = safe_float(patient.get("bmi"))
    bp = safe_float(patient.get("trestbps"))
    chol = safe_float(patient.get("chol"))

    smoking = safe_int(patient.get("smoking"))
    diabetes = safe_int(patient.get("diabetes"))
    family_history = safe_int(
        patient.get("family_history")
    )

    physical_activity = safe_int(
        patient.get("physical_activity")
    )

    stress_level = safe_int(
        patient.get("stress_level")
    )

    # Age
    if age >= 65:
        score -= 15
    elif age >= 55:
        score -= 10
    elif age >= 45:
        score -= 5

    # BMI
    if bmi >= 30:
        score -= 10
    elif bmi >= 25:
        score -= 5
    elif 0 < bmi < 18.5:
        score -= 4

    # Blood pressure
    if bp >= 160:
        score -= 12
    elif bp >= 140:
        score -= 8
    elif bp >= 130:
        score -= 4

    # Cholesterol
    if chol >= 240:
        score -= 10
    elif chol >= 200:
        score -= 5

    # Smoking
    if smoking == 1:
        score -= 10

    # Diabetes
    if diabetes == 1:
        score -= 8

    # Family history
    if family_history == 1:
        score -= 5

    # Physical activity
    if physical_activity == 1:
        score -= 7
    elif physical_activity == 2:
        score -= 2

    # Stress
    if stress_level == 3:
        score -= 7
    elif stress_level == 2:
        score -= 3

    score = max(
        0,
        min(100, score)
    )

    if score >= 75:
        label = "Good"
    elif score >= 50:
        label = "Needs Attention"
    else:
        label = "Needs Improvement"

    return score, label


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(patient):
    recommendations = []

    bmi = safe_float(patient.get("bmi"))
    bp = safe_float(patient.get("trestbps"))
    chol = safe_float(patient.get("chol"))

    smoking = safe_int(patient.get("smoking"))
    diabetes = safe_int(patient.get("diabetes"))
    family_history = safe_int(
        patient.get("family_history")
    )

    physical_activity = safe_int(
        patient.get("physical_activity")
    )

    stress_level = safe_int(
        patient.get("stress_level")
    )

    if smoking == 1:
        recommendations.append(
            "Consider reducing or avoiding tobacco exposure."
        )

    if bmi >= 25:
        recommendations.append(
            "Maintain a balanced diet and regular physical activity."
        )

    if bp >= 140:
        recommendations.append(
            "Keep track of blood pressure and discuss elevated readings with a healthcare professional."
        )

    if chol >= 200:
        recommendations.append(
            "Discuss cholesterol levels with a qualified healthcare professional."
        )

    if physical_activity == 1:
        recommendations.append(
            "Gradually increase regular physical activity as appropriate for your health."
        )

    if stress_level == 3:
        recommendations.append(
            "Consider healthy stress-management practices such as adequate sleep, relaxation, and structured routines."
        )

    if diabetes == 1:
        recommendations.append(
            "Keep diabetes-related health measurements monitored with professional guidance."
        )

    if family_history == 1:
        recommendations.append(
            "Family history can be an important risk context; regular preventive checkups may be useful."
        )

    if not recommendations:
        recommendations.append(
            "Continue maintaining healthy lifestyle habits and regular preventive checkups."
        )

    return recommendations


# ============================================================
# RISK
# ============================================================

def get_risk_level(probability):
    probability = safe_float(probability)

    if probability < 35:
        return "Low", "low"

    if probability < 65:
        return "Moderate", "moderate"

    return "High", "high"


def get_risk_message(risk_level):
    messages = {
        "Low": (
            "The model estimates a lower probability of heart disease "
            "based on the provided parameters."
        ),

        "Moderate": (
            "The model estimates a moderate probability of heart disease. "
            "Consider discussing the result with a qualified healthcare professional."
        ),

        "High": (
            "The model estimates a higher probability of heart disease. "
            "Professional medical evaluation is recommended."
        )
    }

    return messages.get(
        risk_level,
        "Please review the assessment with a healthcare professional."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():
    try:
        importances = model.feature_importances_

        result = dict(
            zip(
                FEATURES,
                importances
            )
        )

        result = dict(
            sorted(
                result.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        return result

    except Exception:
        return {}


# ============================================================
# HISTORY
# ============================================================

def save_history(assessment):
    history = session.get(
        "history",
        []
    )

    history.insert(
        0,
        {
            "patient_name": assessment.get(
                "patient_name",
                "Patient"
            ),

            "risk_level": assessment.get(
                "risk_level",
                "Unknown"
            ),

            "disease_probability": assessment.get(
                "disease_probability",
                0
            ),

            "created_at": datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )
        }
    )

    session["history"] = history[:8]


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "home.html"
    )


# ============================================================
# ASSESSMENT
# ============================================================

@app.route("/assessment")
def assessment():
    # Clear previous assessment data on page load / refresh
    session.pop("assessment", None)
    session.pop("chat_history", None)

    return render_template(
        "index.html",
        prediction_made=False,
        history=session.get("history", [])
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # PROFILE
        # ----------------------------------------------------

        patient_name = request.form.get(
            "patient_name",
            "Patient"
        ).strip()

        age = safe_int(
            request.form.get("age")
        )

        sex = safe_int(
            request.form.get("sex")
        )

        height_cm = safe_float(
            request.form.get("height_cm")
        )

        weight_kg = safe_float(
            request.form.get("weight_kg")
        )

        bmi = calculate_bmi(
            height_cm,
            weight_kg
        )

        # ----------------------------------------------------
        # LIFESTYLE
        # ----------------------------------------------------

        smoking = safe_int(
            request.form.get("smoking")
        )

        diabetes = safe_int(
            request.form.get("diabetes")
        )

        family_history = safe_int(
            request.form.get("family_history")
        )

        physical_activity = safe_int(
            request.form.get("physical_activity")
        )

        stress_level = safe_int(
            request.form.get("stress_level")
        )

        # ----------------------------------------------------
        # CLINICAL FEATURES
        # ----------------------------------------------------

        cp = safe_int(
            request.form.get("cp")
        )

        trestbps = safe_float(
            request.form.get("trestbps")
        )

        chol = safe_float(
            request.form.get("chol")
        )

        fbs = safe_int(
            request.form.get("fbs")
        )

        restecg = safe_int(
            request.form.get("restecg")
        )

        thalach = safe_float(
            request.form.get("thalach")
        )

        exang = safe_int(
            request.form.get("exang")
        )

        oldpeak = safe_float(
            request.form.get("oldpeak")
        )

        slope = safe_int(
            request.form.get("slope")
        )

        ca = safe_int(
            request.form.get("ca")
        )

        thal = safe_int(
            request.form.get("thal")
        )

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]],
            columns=FEATURES
        )

        scaled_input = scaler.transform(
            input_data
        )

        prediction = int(
            model.predict(
                scaled_input
            )[0]
        )

        probabilities = model.predict_proba(
            scaled_input
        )[0]

        no_disease_probability = round(
            float(probabilities[0]) * 100,
            2
        )

        disease_probability = round(
            float(probabilities[1]) * 100,
            2
        )

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        risk_level, risk_class = get_risk_level(
            disease_probability
        )

        if prediction == 1:
            result = "Higher likelihood detected"
        else:
            result = "Lower likelihood detected"

        risk_message = get_risk_message(
            risk_level
        )

        # ----------------------------------------------------
        # HEALTH SCORE
        # ----------------------------------------------------

        patient = {
            "patient_name": patient_name,
            "age": age,
            "sex": sex,

            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,

            "smoking": smoking,
            "diabetes": diabetes,
            "family_history": family_history,

            "physical_activity": physical_activity,
            "stress_level": stress_level,

            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        health_score, health_score_label = calculate_health_score(
            patient
        )

        recommendations = generate_recommendations(
            patient
        )

        feature_importance = get_feature_importance()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        assessment_data = {
            "patient_name": patient_name,

            "patient": patient,

            "prediction": prediction,

            "prediction_made": True,

            "result": result,

            "risk_level": risk_level,

            "risk_class": risk_class,

            "disease_probability": disease_probability,

            "no_disease_probability": no_disease_probability,

            "risk_message": risk_message,

            "health_score": health_score,

            "health_score_label": health_score_label,

            "bmi_category": bmi_category(bmi),

            "recommendations": recommendations,

            "feature_importance": feature_importance
        }

        session["assessment"] = assessment_data

        session["chat_history"] = []

        save_history(
            assessment_data
        )

        return render_template(
            "index.html",
            history=session.get("history", []),
            **assessment_data
        )

    except Exception as e:

        print(
            "PREDICTION ERROR:",
            repr(e)
        )

        return f"""
        <h2>MediRisk AI Error</h2>
        <pre>{str(e)}</pre>
        <br>
        <a href="/assessment?new=1">
            Go back
        </a>
        """, 500


# ============================================================
# AI ASSISTANT
# ============================================================

@app.route(
    "/simulate",
    methods=["POST"]
)
def simulate():
    """Educational what-if simulator using the existing trained model."""
    try:
        assessment = session.get("assessment")
        if not assessment or not assessment.get("patient"):
            return jsonify({"error": "Please complete an assessment first."}), 400

        patient = dict(assessment["patient"])
        payload = request.get_json(silent=True) or {}

        # Only allow model features to be changed in the simulator.
        editable = [
            "trestbps", "chol", "thalach", "oldpeak",
            "cp", "exang", "ca", "thal", "slope",
            "restecg", "fbs", "age", "sex"
        ]

        for key in editable:
            if key in payload and payload[key] not in (None, ""):
                try:
                    value = float(payload[key])
                    patient[key] = int(value) if key in {
                        "age", "sex", "cp", "fbs", "restecg",
                        "exang", "slope", "ca", "thal"
                    } else value
                except (TypeError, ValueError):
                    return jsonify({"error": f"Invalid value for {key}."}), 400

        values = [[patient.get(feature, 0) for feature in FEATURES]]
        scenario_df = pd.DataFrame(values, columns=FEATURES)
        scaled = scaler.transform(scenario_df)
        probs = model.predict_proba(scaled)[0]
        scenario_risk = round(float(probs[1]) * 100, 2)
        current_risk = round(float(assessment.get("disease_probability", 0)), 2)
        change = round(scenario_risk - current_risk, 2)

        return jsonify({
            "current_risk": current_risk,
            "scenario_risk": scenario_risk,
            "change": change,
            "direction": "down" if change < 0 else ("up" if change > 0 else "same")
        })

    except Exception as e:
        print("SIMULATION ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500


# ============================================================
# AI ASSISTANT
# ============================================================

@app.route(
    "/ask-ai",
    methods=["POST"]
)
def ask_ai():

    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get("question", "")
    ).strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    assessment_data = session.get(
        "assessment"
    )

    if not assessment_data:
        return jsonify({
            "error": "Please complete an assessment first."
        }), 400

    if OpenAI is None:
        return jsonify({
            "error": "OpenAI package is not installed."
        }), 500

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        return jsonify({
            "error": "OpenAI API key is not configured."
        }), 500

    try:

        client = OpenAI(
            api_key=api_key
        )

        patient = assessment_data.get(
            "patient",
            {}
        )

        context = f"""
You are MediRisk AI, an educational health assistant.

IMPORTANT:
- You are not a doctor.
- Do not diagnose.
- Do not prescribe medication.
- Do not tell the user to stop medication.
- Explain health concepts in simple language.
- Encourage professional medical evaluation for concerning results.
- Clearly distinguish model prediction from medical diagnosis.

Patient information:
Age: {patient.get("age")}
Sex: {DISPLAY_MAPS["sex"].get(patient.get("sex"), patient.get("sex"))}
BMI: {patient.get("bmi")}
Smoking: {DISPLAY_MAPS["smoking"].get(patient.get("smoking"), patient.get("smoking"))}
Diabetes: {DISPLAY_MAPS["diabetes"].get(patient.get("diabetes"), patient.get("diabetes"))}
Family history: {DISPLAY_MAPS["family_history"].get(patient.get("family_history"), patient.get("family_history"))}
Physical activity: {DISPLAY_MAPS["physical_activity"].get(patient.get("physical_activity"), patient.get("physical_activity"))}
Stress: {DISPLAY_MAPS["stress_level"].get(patient.get("stress_level"), patient.get("stress_level"))}

Clinical parameters:
Chest pain: {DISPLAY_MAPS["cp"].get(patient.get("cp"), patient.get("cp"))}
Resting BP: {patient.get("trestbps")}
Cholesterol: {patient.get("chol")}
Fasting blood sugar: {DISPLAY_MAPS["fbs"].get(patient.get("fbs"), patient.get("fbs"))}
Resting ECG: {DISPLAY_MAPS["restecg"].get(patient.get("restecg"), patient.get("restecg"))}
Maximum heart rate: {patient.get("thalach")}
Exercise angina: {DISPLAY_MAPS["exang"].get(patient.get("exang"), patient.get("exang"))}
Oldpeak: {patient.get("oldpeak")}
Slope: {DISPLAY_MAPS["slope"].get(patient.get("slope"), patient.get("slope"))}
Major vessels: {patient.get("ca")}
Thalassemia: {DISPLAY_MAPS["thal"].get(patient.get("thal"), patient.get("thal"))}

MediRisk AI model result:
Risk level: {assessment_data.get("risk_level")}
Disease probability: {assessment_data.get("disease_probability")}%
No disease probability: {assessment_data.get("no_disease_probability")}%
Educational health score: {assessment_data.get("health_score")}/100

User question:
{question}

Answer in a concise, friendly and educational way.
"""

        chat_history = session.get("chat_history", [])
        recent_conversation = "\n".join(
            f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
            for item in chat_history[-8:]
        )
        if recent_conversation:
            context += f"""
Recent conversation:
{recent_conversation}

Use this only for follow-up context. Do not treat previous AI responses as medical facts.
"""

        response = client.responses.create(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-5.6-luna"
            ),
            input=context
        )

        answer = response.output_text

        chat_history = session.get(
            "chat_history",
            []
        )

        chat_history.append({
            "role": "user",
            "content": question
        })

        chat_history.append({
            "role": "assistant",
            "content": answer
        })

        session["chat_history"] = chat_history[-12:]

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print(
            "AI ERROR:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# CLEAR AI CHAT
# ============================================================

@app.route("/clear-chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    return jsonify({"success": True})


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    session["history"] = []

    return jsonify({
        "success": True
    })


# ============================================================
# DOWNLOAD PDF
# ============================================================

@app.route("/download-report")
def download_report():
    """Generate a polished two-page MediRisk AI assessment report."""
    assessment_data = session.get("assessment")

    if not assessment_data:
        return "No assessment available.", 404

    if not REPORTLAB_AVAILABLE:
        return (
            "ReportLab is not installed. Run: pip install reportlab"
        ), 500

    patient = assessment_data.get("patient", {})

    # ------------------------------------------------------------
    # ReportLab imports used only by the PDF route
    # ------------------------------------------------------------
    from html import escape
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.pdfbase.pdfmetrics import stringWidth

    # Premium MediRisk palette
    NAVY = colors.HexColor("#071126")
    NAVY_2 = colors.HexColor("#0B1733")
    BLUE = colors.HexColor("#3B82F6")
    PURPLE = colors.HexColor("#8B5CF6")
    CYAN = colors.HexColor("#22D3EE")
    TEXT = colors.HexColor("#172033")
    MUTED = colors.HexColor("#667085")
    LIGHT = colors.HexColor("#F5F7FB")
    BORDER = colors.HexColor("#D9E1EF")
    WHITE = colors.white
    GREEN = colors.HexColor("#10B981")
    AMBER = colors.HexColor("#F59E0B")
    RED = colors.HexColor("#EF4444")

    risk_level = str(assessment_data.get("risk_level", "Unknown"))
    risk_probability = safe_float(assessment_data.get("disease_probability", 0))
    no_disease_probability = safe_float(
        assessment_data.get("no_disease_probability", 0)
    )

    if risk_level.lower() == "low":
        risk_color = GREEN
    elif risk_level.lower() == "moderate":
        risk_color = AMBER
    else:
        risk_color = RED

    patient_name = str(patient.get("patient_name") or assessment_data.get("patient_name") or "Patient")
    generated_at = datetime.now().strftime("%d %B %Y • %I:%M %p")

    # ------------------------------------------------------------
    # Document + page chrome
    # ------------------------------------------------------------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=25 * mm,
        bottomMargin=18 * mm,
        title="MediRisk AI — Heart Disease Risk Assessment Report",
        author="MediRisk AI",
        subject="Educational heart disease risk assessment",
    )

    page_width, page_height = A4

    def draw_page_chrome(canvas, doc_obj):
        canvas.saveState()

        # Header band
        canvas.setFillColor(NAVY)
        canvas.rect(0, page_height - 20 * mm, page_width, 20 * mm, fill=1, stroke=0)

        # Purple/blue accent line
        canvas.setFillColor(PURPLE)
        canvas.rect(0, page_height - 20 * mm, page_width * 0.62, 1.2 * mm, fill=1, stroke=0)
        canvas.setFillColor(BLUE)
        canvas.rect(page_width * 0.62, page_height - 20 * mm, page_width * 0.38, 1.2 * mm, fill=1, stroke=0)

        # Brand mark
        canvas.setFillColor(PURPLE)
        canvas.roundRect(14 * mm, page_height - 15.8 * mm, 9 * mm, 9 * mm, 2.2 * mm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(18.5 * mm, page_height - 13.5 * mm, "♥")

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(26 * mm, page_height - 11.8 * mm, "MediRisk AI")
        canvas.setFillColor(colors.HexColor("#B9C5E3"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(26 * mm, page_height - 15.4 * mm, "Intelligent Health Assessment")

        # Right header label
        canvas.setFillColor(colors.HexColor("#C8D4F5"))
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawRightString(page_width - 14 * mm, page_height - 12.2 * mm, "EDUCATIONAL AI REPORT")
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(page_width - 14 * mm, page_height - 15.4 * mm, generated_at)

        # Footer
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(14 * mm, 12 * mm, page_width - 14 * mm, 12 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6.8)
        canvas.drawString(14 * mm, 8 * mm, "MediRisk AI • Educational & research-oriented machine learning application")
        canvas.drawRightString(page_width - 14 * mm, 8 * mm, f"Page {doc_obj.page}")

        canvas.restoreState()

    # ------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=25,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=MUTED,
        spaceAfter=13,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.4,
        leading=12,
        textColor=TEXT,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=7.3,
        leading=10,
        textColor=MUTED,
    )

    value_style = ParagraphStyle(
        "Value",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=NAVY,
    )

    story = []

    # ------------------------------------------------------------
    # PAGE 1 — Executive summary
    # ------------------------------------------------------------
    story.append(Paragraph("Heart Disease Risk Assessment", title_style))
    story.append(Paragraph(
        "A structured summary of the patient's submitted parameters and the MediRisk AI model output.",
        subtitle_style,
    ))

    # Patient identity card
    patient_card = Table([
        [
            Paragraph("PATIENT", small_style),
            Paragraph("ASSESSMENT DATE", small_style),
            Paragraph("SYSTEM", small_style),
        ],
        [
            Paragraph(escape(patient_name), value_style),
            Paragraph(escape(generated_at), value_style),
            Paragraph("MediRisk AI • Random Forest", value_style),
        ],
    ], colWidths=[57 * mm, 60 * mm, 58 * mm])
    patient_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_card)
    story.append(Spacer(1, 10))

    # ------------------------------------------------------------
    # Premium visual risk panel — circular score + ECG line
    # ------------------------------------------------------------
    from reportlab.pdfgen import canvas as pdf_canvas

    class RiskVisualPanel(Flowable):
        def __init__(self, probability, risk_label, risk_fill, health_score):
            Flowable.__init__(self)
            self.probability = max(0.0, min(100.0, float(probability)))
            self.risk_label = str(risk_label)
            self.risk_fill = risk_fill
            self.health_score = max(0, min(100, int(safe_float(health_score))))
            self.width = 175 * mm
            self.height = 48 * mm

        def draw(self):
            c = self.canv
            w, h = self.width, self.height

            # Panel
            c.saveState()
            c.setFillColor(colors.HexColor("#F7F9FC"))
            c.setStrokeColor(BORDER)
            c.setLineWidth(0.7)
            c.roundRect(0, 0, w, h, 4 * mm, fill=1, stroke=1)

            # Accent strip
            c.setFillColor(self.risk_fill)
            c.roundRect(0, h - 2.2 * mm, w, 2.2 * mm, 1.1 * mm, fill=1, stroke=0)

            # Circular risk gauge
            cx = 27 * mm
            cy = h / 2
            radius = 15 * mm

            c.setStrokeColor(colors.HexColor("#E3E8F2"))
            c.setLineWidth(5)
            c.circle(cx, cy, radius, stroke=1, fill=0)

            c.setStrokeColor(self.risk_fill)
            c.setLineWidth(5)
            c.setLineCap(1)
            c.arc(cx - radius, cy - radius, cx + radius, cy + radius,
                  startAng=90, extent=-360 * (self.probability / 100.0))

            c.setFillColor(NAVY)
            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(cx, cy + 1.5 * mm, f"{self.probability:.1f}%")
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 6.8)
            c.drawCentredString(cx, cy - 5 * mm, "DISEASE RISK")

            # Risk text
            text_x = 49 * mm
            c.setFillColor(MUTED)
            c.setFont("Helvetica-Bold", 6.8)
            c.drawString(text_x, h - 12 * mm, "AI RISK ASSESSMENT")

            c.setFillColor(self.risk_fill)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(text_x, h - 20 * mm, f"{self.risk_label} Risk")

            c.setFillColor(TEXT)
            c.setFont("Helvetica", 7.6)
            c.drawString(text_x, h - 27 * mm, "Model probability based on submitted clinical parameters")

            # Health score badge
            bx = 113 * mm
            by = h - 28 * mm
            c.setFillColor(WHITE)
            c.setStrokeColor(BORDER)
            c.roundRect(bx, by, 26 * mm, 16 * mm, 3 * mm, fill=1, stroke=1)

            c.setFillColor(MUTED)
            c.setFont("Helvetica-Bold", 6.3)
            c.drawCentredString(bx + 13 * mm, by + 11 * mm, "HEALTH SCORE")
            c.setFillColor(NAVY)
            c.setFont("Helvetica-Bold", 11)
            c.drawCentredString(bx + 13 * mm, by + 4.5 * mm, f"{self.health_score}/100")

            # ECG line
            ecg_left = 145 * mm
            ecg_right = w - 7 * mm
            base = 13 * mm
            points = [
                (ecg_left, base),
                (ecg_left + 7*mm, base),
                (ecg_left + 10*mm, base + 2*mm),
                (ecg_left + 13*mm, base - 1*mm),
                (ecg_left + 16*mm, base),
                (ecg_left + 20*mm, base + 11*mm),
                (ecg_left + 23*mm, base - 8*mm),
                (ecg_left + 27*mm, base + 3*mm),
                (ecg_left + 31*mm, base),
                (ecg_left + 38*mm, base),
            ]

            c.setStrokeColor(colors.HexColor("#D9E1EF"))
            c.setLineWidth(0.6)
            c.line(ecg_left, base, ecg_right, base)

            c.setStrokeColor(self.risk_fill)
            c.setLineWidth(1.5)
            path = c.beginPath()
            path.moveTo(*points[0])
            for px, py in points[1:]:
                path.lineTo(px, py)
            c.drawPath(path, stroke=1, fill=0)

            c.setFillColor(MUTED)
            c.setFont("Helvetica", 6)
            c.drawString(ecg_left, 6 * mm, "HEART SIGNAL • EDUCATIONAL VISUAL")

            c.restoreState()

    story.append(
        RiskVisualPanel(
            risk_probability,
            risk_level,
            risk_color,
            assessment_data.get("health_score", 0),
        )
    )
    story.append(Spacer(1, 9))

    # Risk summary card
    risk_summary = Table([
        [
            Paragraph("MODEL ASSESSMENT", small_style),
            Paragraph("RISK CATEGORY", small_style),
            Paragraph("DISEASE PROBABILITY", small_style),
            Paragraph("NO DISEASE", small_style),
        ],
        [
            Paragraph(escape(str(assessment_data.get("result", "Assessment generated"))), value_style),
            Paragraph(f'<font color="{risk_color.hexval()}"><b>{escape(risk_level)} Risk</b></font>', value_style),
            Paragraph(f'<font color="{risk_color.hexval()}"><b>{risk_probability:.2f}%</b></font>', value_style),
            Paragraph(f"<b>{no_disease_probability:.2f}%</b>", value_style),
        ],
    ], colWidths=[60 * mm, 38 * mm, 40 * mm, 37 * mm])
    risk_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.9, risk_color),
        ("LINEBEFORE", (1, 0), (1, -1), 0.35, BORDER),
        ("LINEBEFORE", (2, 0), (2, -1), 0.35, BORDER),
        ("LINEBEFORE", (3, 0), (3, -1), 0.35, BORDER),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(risk_summary)
    story.append(Spacer(1, 8))

    # Risk interpretation band
    risk_message = str(assessment_data.get("risk_message", "Review this educational result with a qualified healthcare professional."))
    interpretation = Table([
        [
            Paragraph(
                f'<font color="{risk_color.hexval()}"><b>Risk interpretation</b></font><br/>{escape(risk_message)}',
                body_style,
            )
        ]
    ], colWidths=[175 * mm])
    interpretation.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFE")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(interpretation)

    story.append(Paragraph("Patient Profile", section_style))
    profile_rows = [
        ["Age", f"{patient.get('age', '-') } years", "Sex", DISPLAY_MAPS["sex"].get(patient.get("sex"), "-")],
        ["Height", f"{patient.get('height_cm', '-')} cm", "Weight", f"{patient.get('weight_kg', '-')} kg"],
        ["BMI", f"{patient.get('bmi', '-')} ({assessment_data.get('bmi_category', '-')})", "Health Score", f"{assessment_data.get('health_score', 0)}/100 • {assessment_data.get('health_score_label', '-') }"],
    ]
    profile_table = Table(profile_rows, colWidths=[28 * mm, 60 * mm, 30 * mm, 57 * mm])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(profile_table)

    story.append(Paragraph("Clinical Parameters", section_style))
    clinical_rows = [
        ["Chest Pain", DISPLAY_MAPS["cp"].get(patient.get("cp"), "-"), "Resting BP", f"{patient.get('trestbps', '-')} mm Hg"],
        ["Cholesterol", f"{patient.get('chol', '-')} mg/dL", "Fasting Blood Sugar", DISPLAY_MAPS["fbs"].get(patient.get("fbs"), "-")],
        ["Resting ECG", DISPLAY_MAPS["restecg"].get(patient.get("restecg"), "-"), "Maximum Heart Rate", f"{patient.get('thalach', '-')} bpm"],
        ["Exercise Angina", DISPLAY_MAPS["exang"].get(patient.get("exang"), "-"), "ST Depression", str(patient.get("oldpeak", "-"))],
        ["ST Slope", DISPLAY_MAPS["slope"].get(patient.get("slope"), "-"), "Major Vessels", str(patient.get("ca", "-"))],
        ["Thalassemia", DISPLAY_MAPS["thal"].get(patient.get("thal"), "-"), "Model", "Random Forest"],
    ]
    clinical_table = Table(clinical_rows, colWidths=[38 * mm, 50 * mm, 42 * mm, 45 * mm])
    clinical_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("FONTSIZE", (0, 0), (-1, -1), 7.7),
        ("TOPPADDING", (0, 0), (-1, -1), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(clinical_table)

    # ------------------------------------------------------------
    # PAGE 2 — Lifestyle, model factors, recommendations
    # ------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Supporting Health Insights", title_style))
    story.append(Paragraph(
        "Additional lifestyle context, model factors, and educational recommendations associated with this assessment.",
        subtitle_style,
    ))

    # Lifestyle profile
    story.append(Paragraph("Lifestyle & Health Profile", section_style))
    lifestyle_rows = [
        ["Smoking", DISPLAY_MAPS["smoking"].get(patient.get("smoking"), "-"), "Diabetes", DISPLAY_MAPS["diabetes"].get(patient.get("diabetes"), "-")],
        ["Family History", DISPLAY_MAPS["family_history"].get(patient.get("family_history"), "-"), "Physical Activity", DISPLAY_MAPS["physical_activity"].get(patient.get("physical_activity"), "-")],
        ["Stress Level", DISPLAY_MAPS["stress_level"].get(patient.get("stress_level"), "-"), "BMI Category", assessment_data.get("bmi_category", "-")],
    ]
    lifestyle_table = Table(lifestyle_rows, colWidths=[40 * mm, 48 * mm, 42 * mm, 45 * mm])
    lifestyle_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(lifestyle_table)
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Lifestyle fields are recorded as additional patient context. They are not direct inputs to the existing 13-feature Random Forest prediction model.",
        small_style,
    ))

    # Model information strip
    model_strip = Table([
        [
            Paragraph("<b>MODEL</b><br/>Random Forest", small_style),
            Paragraph("<b>ACCURACY</b><br/>86.67%", small_style),
            Paragraph("<b>F1 SCORE</b><br/>85.19%", small_style),
            Paragraph("<b>ROC-AUC</b><br/>94.20%", small_style),
        ]
    ], colWidths=[43.75 * mm] * 4)
    model_strip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FC")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(model_strip)
    story.append(Spacer(1, 7))

    # Probability visual
    story.append(Paragraph("Prediction Probability", section_style))
    bar_width = 132 * mm
    disease_width = max(1, bar_width * min(risk_probability, 100) / 100)
    no_disease_width = max(1, bar_width * min(no_disease_probability, 100) / 100)

    def probability_bar(label, value, fill_color):
        bar = Table([["", ""]], colWidths=[bar_width * min(value, 100) / 100, bar_width * max(0, 100 - min(value, 100)) / 100])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), fill_color),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E8EDF6")),
            ("BOX", (0, 0), (-1, -1), 0, colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        row = Table([
            [Paragraph(f"<b>{escape(label)}</b>", small_style), Paragraph(f"<b>{value:.2f}%</b>", ParagraphStyle("right", parent=small_style, alignment=TA_RIGHT))],
            [bar, ""],
        ], colWidths=[140 * mm, 35 * mm])
        row.setStyle(TableStyle([
            ("SPAN", (0, 1), (1, 1)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return row

    story.append(probability_bar("Disease Risk", risk_probability, risk_color))
    story.append(probability_bar("No Disease", no_disease_probability, GREEN))

    # Model factors
    story.append(Paragraph("Top Model Factors", section_style))
    importance = assessment_data.get("feature_importance") or {}
    factor_rows = [["Feature", "Relative Importance"]]
    for feature, score in list(importance.items())[:7]:
        factor_rows.append([
            feature.replace("_", " ").title(),
            f"{safe_float(score) * 100:.2f}%",
        ])

    factor_table = Table(factor_rows, colWidths=[125 * mm, 50 * mm])
    factor_style = [
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]
    for r in range(1, len(factor_rows)):
        if r % 2 == 0:
            factor_style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    factor_table.setStyle(TableStyle(factor_style))
    story.append(factor_table)

    # Recommendations
    story.append(Paragraph("Educational Recommendations", section_style))
    recommendations = assessment_data.get("recommendations") or []
    rec_rows = []
    for rec in recommendations[:8]:
        rec_rows.append([
            Paragraph('<font color="#8B5CF6"><b>•</b></font>', body_style),
            Paragraph(escape(str(rec)), body_style),
        ])
    if not rec_rows:
        rec_rows.append([
            Paragraph('<font color="#8B5CF6"><b>•</b></font>', body_style),
            Paragraph("Continue maintaining healthy lifestyle habits and regular preventive checkups.", body_style),
        ])
    rec_table = Table(rec_rows, colWidths=[7 * mm, 168 * mm])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#E7EBF3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(rec_table)

    # Methodology + disclaimer
    story.append(Paragraph("Assessment Notes", section_style))
    notes = Table([
        [Paragraph(
            "<b>Model:</b> Random Forest &nbsp;&nbsp; <b>Assessment type:</b> Binary heart disease risk prediction<br/>"
            "The model output is a probability estimate based on the submitted clinical parameters. It is not a diagnosis.",
            body_style,
        )],
        [Paragraph(
            '<font color="#B45309"><b>Important Medical Disclaimer</b></font><br/>'
            "MediRisk AI is designed for educational and research purposes. The prediction, probability, health score, "
            "and recommendations should not be considered medical diagnosis, treatment advice, or a substitute for "
            "evaluation by a qualified healthcare professional. Seek professional care for symptoms or concerns.",
            body_style,
        )],
    ], colWidths=[175 * mm])
    notes.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F8FAFD")),
        ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#FFF8E8")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("LINEBELOW", (0, 0), (0, 0), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(notes)
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        "Generated by MediRisk AI • Educational AI Health Assessment",
        ParagraphStyle("Generated", parent=small_style, alignment=TA_RIGHT),
    ))

    doc.build(story, onFirstPage=draw_page_chrome, onLaterPages=draw_page_chrome)

    buffer.seek(0)
    safe_filename = "MediRisk_AI_Health_Report.pdf"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=safe_filename,
        mimetype="application/pdf",
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "online",
        "model": "Random Forest",
        "scaler": "loaded",
        "ai_configured": bool(
            os.getenv("OPENAI_API_KEY")
        )
    })




# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("MediRisk AI")
    print("=" * 55)
    print("Model: Random Forest")
    print("Server: http://127.0.0.1:5000")
    print("=" * 55)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )