import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

HISTORY_FILE = "prediction_history.csv"

# ---------------------------------------------------------
# Custom CSS — professional clinical / calm blue theme
# (works together with .streamlit/config.toml which forces
#  the light base theme so widgets aren't dark-on-dark)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #EAF2FB 0%, #F6FAFE 100%);
    }

    .clinic-header {
        background: linear-gradient(135deg, #1B4F91 0%, #2E6FBF 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white !important;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(27, 79, 145, 0.25);
    }
    .clinic-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 700;
        color: white !important;
    }
    .clinic-header p {
        margin: 6px 0 0 0;
        font-size: 15px;
        opacity: 0.9;
        color: white !important;
    }

    /* Style Streamlit's NATIVE bordered container as our "card" */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border-radius: 16px !important;
        border: 1px solid #DCE8F7 !important;
        box-shadow: 0 4px 18px rgba(27, 79, 145, 0.08);
        padding: 6px 8px;
        margin-bottom: 20px;
    }

    .step-title {
        font-size: 20px;
        font-weight: 700;
        color: #1B4F91 !important;
        margin-bottom: 4px;
    }
    .step-subtitle {
        font-size: 13px;
        color: #6B87A8 !important;
        margin-bottom: 20px;
    }

    .progress-wrap {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 24px;
    }
    .dot {
        width: 34px;
        height: 6px;
        border-radius: 4px;
        background: #C9DBF0;
        transition: all 0.3s ease;
    }
    .dot.active {
        background: #1B4F91;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 22px;
        border: none;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        width: 100%;
    }

    /* Force readable labels & widget text regardless of system theme */
    label, .stMarkdown p, .stMarkdown li,
    div[data-testid="stWidgetLabel"] p {
        color: #1E3A5F !important;
    }

    /* Text/number inputs, selects, sliders — light surfaces */
    div[data-baseweb="input"], div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        color: #1E3A5F !important;
        border: 1px solid #C9DBF0 !important;
    }
    input, textarea {
        color: #1E3A5F !important;
        background-color: #FFFFFF !important;
    }

    .result-safe {
        background: #E4F7EC;
        border: 1px solid #8FD9AE;
        color: #1E7A46;
        padding: 20px 24px;
        border-radius: 14px;
        font-size: 16px;
        font-weight: 600;
    }
    .result-risk {
        background: #FDEBEC;
        border: 1px solid #F0A6AB;
        color: #B3272D;
        padding: 20px 24px;
        border-radius: 14px;
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="clinic-header">
    <h1>🫀 Heart Disease Risk Assessment</h1>
    <p>A step-by-step clinical screening tool by Ayan</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session state setup
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "data" not in st.session_state:
    st.session_state.data = {}
if "saved_this_result" not in st.session_state:
    st.session_state.saved_this_result = False

TOTAL_STEPS = 4

def progress_dots():
    dots = ""
    for i in range(1, TOTAL_STEPS + 1):
        cls = "dot active" if i <= st.session_state.step else "dot"
        dots += f'<div class="{cls}"></div>'
    st.markdown(f'<div class="progress-wrap">{dots}</div>', unsafe_allow_html=True)

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

progress_dots()

# ===========================================================
# STEP 1 — Patient Info
# ===========================================================
if st.session_state.step == 1:
    with st.container(border=True):
        st.markdown('<div class="step-title">👤 Patient Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-subtitle">Basic personal information</div>', unsafe_allow_html=True)

        name = st.text_input("📝 Patient Name", st.session_state.data.get("name", ""))
        age = st.slider("🎂 Age", 18, 100, st.session_state.data.get("age", 40))
        sex = st.selectbox("⚧ Sex", ["M", "F"], index=["M", "F"].index(st.session_state.data.get("sex", "M")))

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next →", type="primary"):
            if not name.strip():
                st.warning("Please enter the patient's name to continue.")
            else:
                st.session_state.data.update({"name": name.strip(), "age": age, "sex": sex})
                next_step()
                st.rerun()

# ===========================================================
# STEP 2 — Clinical Vitals
# ===========================================================
elif st.session_state.step == 2:
    with st.container(border=True):
        st.markdown('<div class="step-title">🩺 Clinical Vitals</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-subtitle">Blood pressure, cholesterol & related readings</div>', unsafe_allow_html=True)

        resting_bp = st.number_input("💉 Resting Blood Pressure (mm Hg)", 80, 200,
                                      st.session_state.data.get("resting_bp", 120))
        cholesterol = st.number_input("🧪 Cholesterol (mg/dl)", 100, 600,
                                       st.session_state.data.get("cholesterol", 200))
        fasting_bs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                   index=[0, 1].index(st.session_state.data.get("fasting_bs", 0)))
        resting_ecg = st.selectbox("📈 Resting ECG", ["Normal", "ST", "LVH"],
                                    index=["Normal", "ST", "LVH"].index(st.session_state.data.get("resting_ecg", "Normal")))
        max_hr = st.slider("❤️ Max Heart Rate", 60, 220, st.session_state.data.get("max_hr", 150))

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.data.update({
                "resting_bp": resting_bp, "cholesterol": cholesterol,
                "fasting_bs": fasting_bs, "resting_ecg": resting_ecg, "max_hr": max_hr
            })
            next_step()
            st.rerun()

# ===========================================================
# STEP 3 — Symptoms
# ===========================================================
elif st.session_state.step == 3:
    with st.container(border=True):
        st.markdown('<div class="step-title">⚠️ Symptoms & Test Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-subtitle">Chest pain type and exercise-related findings</div>', unsafe_allow_html=True)

        chest_pain = st.selectbox("💢 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
                                   index=["ATA", "NAP", "TA", "ASY"].index(st.session_state.data.get("chest_pain", "ATA")))
        exercise_angina = st.selectbox("🏃 Exercise Induced Angina", ["N", "Y"],
                                        index=["N", "Y"].index(st.session_state.data.get("exercise_angina", "N")))
        oldpeak = st.slider("📉 Oldpeak (ST depression)", 0.0, 6.0, st.session_state.data.get("oldpeak", 1.0))
        st_slope = st.selectbox("📊 ST Slope", ["Up", "Flat", "Down"],
                                 index=["Up", "Flat", "Down"].index(st.session_state.data.get("st_slope", "Up")))

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Get Results →", type="primary"):
            st.session_state.data.update({
                "chest_pain": chest_pain, "exercise_angina": exercise_angina,
                "oldpeak": oldpeak, "st_slope": st_slope
            })
            st.session_state.saved_this_result = False
            next_step()
            st.rerun()

# ===========================================================
# STEP 4 — Results + History
# ===========================================================
elif st.session_state.step == 4:
    d = st.session_state.data

    raw_input = {
        'Age': d["age"],
        'RestingBP': d["resting_bp"],
        'Cholesterol': d["cholesterol"],
        'FastingBS': d["fasting_bs"],
        'MaxHR': d["max_hr"],
        'Oldpeak': d["oldpeak"],
        'Sex_' + d["sex"]: 1,
        'ChestPainType_' + d["chest_pain"]: 1,
        'RestingECG_' + d["resting_ecg"]: 1,
        'ExerciseAngina_' + d["exercise_angina"]: 1,
        'ST_Slope_' + d["st_slope"]: 1,
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    try:
        proba = model.predict_proba(scaled_input)[0][1] * 100
    except Exception:
        proba = 85 if prediction == 1 else 15

    name = d.get("name", "Patient")

    with st.container(border=True):
        st.markdown('<div class="step-title">📋 Assessment Result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="step-subtitle">Result for {name}, age {d["age"]}</div>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            number={'suffix': "%", 'font': {'size': 40, 'color': '#1B4F91'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#1B4F91'},
                'bar': {'color': '#1B4F91'},
                'steps': [
                    {'range': [0, 40], 'color': '#DFF5E7'},
                    {'range': [40, 70], 'color': '#FFF3D0'},
                    {'range': [70, 100], 'color': '#FBE0E1'}
                ],
                'threshold': {
                    'line': {'color': '#B3272D', 'width': 4},
                    'thickness': 0.8,
                    'value': proba
                }
            },
            title={'text': "Estimated Risk Level", 'font': {'size': 16, 'color': '#1E3A5F'}}
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-risk">
            ⚠️ {name}, the model predicts a <b>likely presence of heart disease</b>.
            Please consult a healthcare professional for further evaluation.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
            ✅ {name}, the model predicts that heart disease is <b>unlikely</b> based on the provided data.
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # Save this result to history (once per generated result)
    # -------------------------------------------------------
    if not st.session_state.saved_this_result:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "age": d["age"],
            "sex": d["sex"],
            "risk_percent": round(proba, 1),
            "prediction": "At Risk" if prediction == 1 else "Low Risk",
        }
        file_exists = os.path.isfile(HISTORY_FILE)
        pd.DataFrame([record]).to_csv(HISTORY_FILE, mode="a", header=not file_exists, index=False)
        st.session_state.saved_this_result = True

    # -------------------------------------------------------
    # History section
    # -------------------------------------------------------
    if os.path.isfile(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)

        with st.container(border=True):
            st.markdown('<div class="step-title">📊 Prediction History</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-subtitle">Risk trend across all past assessments</div>', unsafe_allow_html=True)

            chart_df = history_df.copy()
            chart_df["record"] = range(1, len(chart_df) + 1)

            hist_fig = px.line(
                chart_df, x="record", y="risk_percent", markers=True,
                hover_data=["name", "timestamp", "prediction"],
                color_discrete_sequence=["#1B4F91"],
            )
            hist_fig.update_traces(line=dict(width=3), marker=dict(size=8, color="#2E6FBF"))
            hist_fig.update_layout(
                height=300,
                margin=dict(t=20, b=10, l=10, r=10),
                xaxis_title="Assessment #",
                yaxis_title="Risk %",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#1E3A5F"),
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            with st.expander("View raw history table"):
                st.dataframe(history_df.sort_values("timestamp", ascending=False), use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("🔄 Start Over", type="primary"):
            st.session_state.step = 1
            st.session_state.data = {}
            st.session_state.saved_this_result = False
            st.rerun()