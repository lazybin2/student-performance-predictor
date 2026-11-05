import streamlit as st
import numpy as np
import pandas as pd
from model import train_student_model, predict_performance, generate_recommendations

# Page Configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Advanced Student Performance Predictor (Random Forest)")
st.write("An intelligent exam result prediction system featuring consistent decision boundaries, risk analysis, and visual metrics.")

@st.cache_resource
def get_model():
    return train_student_model()

model, scaler = get_model()

col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📝 Provide Student Information:")
    
    study_hours = st.slider("Daily Study Hours", 0.0, 12.0, 4.0)
    attendance = st.slider("Class Attendance (%)", 0.0, 100.0, 70.0)
    prev_gpa = st.slider("Previous CGPA (GPA)", 0.0, 4.0, 2.8)
    sleep_hours = st.slider("Daily Sleep Hours", 0.0, 12.0, 6.5)
    extra_activity = st.selectbox("Extracurricular Activities", ["No", "Yes"])

    extra_val = 1 if extra_activity == "Yes" else 0
    predict_btn = st.button("🔍 Predict Performance", use_container_width=True)

with col_output:
    st.subheader("📊 Analysis & Prediction Results:")
    
    if predict_btn:
        input_data = np.array([[study_hours, attendance / 100.0, prev_gpa / 4.0, sleep_hours, extra_val]])
        probability = predict_performance(model, scaler, input_data)
        
        # Display Result
        if probability >= 0.5:
            st.success(f"🎉 Result: **PASS** (Confidence: {probability * 100:.2f}%)")
        else:
            st.error(f"⚠️ Result: **FAIL / AT RISK** (Risk Score: {(1 - probability) * 100:.2f}%)")
            
        # Visual Metrics Comparison Chart
        st.markdown("### 📈 Visual Metrics Comparison")
        chart_data = pd.DataFrame({
            "Normalized Value": [study_hours / 12.0, attendance / 100.0, prev_gpa / 4.0, sleep_hours / 12.0]
        }, index=["Study Hours", "Attendance", "GPA", "Sleep Hours"])
        
        st.bar_chart(chart_data)

        # Risk Factor Breakdown Progress Bars
        st.markdown("### 🔍 Risk Factor Breakdown")
        st.progress(int(attendance), text=f"Attendance Impact: {attendance}%")
        st.progress(int((study_hours / 12.0) * 100), text=f"Study Commitment: {study_hours} Hours/Day")
        st.progress(int((prev_gpa / 4.0) * 100), text=f"Academic Background (GPA): {prev_gpa}")

        # AI Recommendations & Tips (Now passing the probability status)
        st.markdown("### 💡 AI Recommendations & Tips")
        tips = generate_recommendations(study_hours, attendance, prev_gpa, sleep_hours, probability)
        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        st.info("👈 Please input student metrics on the left panel and click 'Predict Performance'.")