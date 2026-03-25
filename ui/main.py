import streamlit as st
import requests

from config import API_URI
from api_utils import APICalls


api = APICalls(API_URI)

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# --- HEADER ---
st.markdown(
    """
    <h1 style='text-align: center;'>❤️ Heart Disease Risk Predictor</h1>
    <p style='text-align: center; color: gray;'>
        Enter patient data to estimate cardiovascular risk
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Create two main columns: left for inputs, right for prediction
left_col, right_col = st.columns([1, 1])

with left_col:
    # --- FORM ---
    with st.form("input_form"):

        # Use 2 main columns for layout
        left, right = st.columns([1, 1])

        # ---------------- LEFT COLUMN ----------------
        with left:
            st.markdown("### 👤 Demographics")
            col1, col2 = st.columns(2)

            with col1:
                male = st.selectbox("Gender", ["male", "female"])
                age = st.slider("Age", 0, 120, 30)

            with col2:
                education = st.selectbox("Education Level", ["1", "2", "3", "4"])

            st.markdown("### 🚬 Smoking")
            col1, col2 = st.columns(2)

            with col1:
                currentSmoker = st.radio("Current Smoker", ["yes", "no"], horizontal=True)

            with col2:
                cigsPerDay = st.slider(
                    "Cigarettes/day",
                    0,
                    120,
                    0,
                    disabled=(currentSmoker == "no")
                )

            st.markdown("### 🏥 Medical History")
            col1, col2 = st.columns(2)

            with col1:
                BPMeds = st.number_input("BP Medications", min_value=0.0, value=0.0)
                prevalentStroke = st.radio("Stroke", ["yes", "no"], horizontal=True)

            with col2:
                prevalentHyp = st.radio("Hypertension", ["yes", "no"], horizontal=True)
                diabetes = st.radio("Diabetes", ["yes", "no"], horizontal=True)

        # ---------------- RIGHT COLUMN ----------------
        with right:
            st.markdown("### 📊 Health Metrics")

            col1, col2 = st.columns(2)

            with col1:
                totChol = st.number_input("Cholesterol", 0.0, value=200.0)
                sysBP = st.number_input("Systolic BP", 0.0, value=120.0)
                BMI = st.number_input("BMI", 0.0, value=25.0)

            with col2:
                diaBP = st.number_input("Diastolic BP", 0.0, value=80.0)
                heartRate = st.number_input("Heart Rate", 0.0, value=70.0)
                glucose = st.number_input("Glucose", 0.0, value=90.0)

        st.divider()

        submit = st.form_submit_button("🔍 Predict Risk")

with right_col:

    # --- ON SUBMIT ---
    if submit:
        payload = {
            "male": male,
            "age": age,
            "education": education,
            "currentSmoker": currentSmoker,
            "cigsPerDay": cigsPerDay,
            "BPMeds": BPMeds,
            "prevalentStroke": prevalentStroke,
            "prevalentHyp": prevalentHyp,
            "diabetes": diabetes,
            "totChol": totChol,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "BMI": BMI,
            "heartRate": heartRate,
            "glucose": glucose,
        }

        st.write("### Input Data")
        st.json(payload, expanded=False)

        # --- CALL API (adjust URL later) ---
        try:

            result = api.predict_proba(payload)
            prediction = result["prediction"]

            st.divider()
            st.subheader("🧠 Prediction Result")

            # --- RISK VISUALIZATION ---
            st.metric(
                label="Predicted Risk",
                value=f"{prediction*100:.2f}%",
            )

            st.progress(float(prediction))

            # --- INTERPRETATION ---
            if prediction < 0.3:
                st.success("🟢 Low risk")
            elif prediction < 0.7:
                st.warning("🟡 Moderate risk")
            else:
                st.error("🔴 High risk")

        except Exception as e:
            st.error(f"Connection error: {e}")
