import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

@st.cache_resource
def train_student_model():
    print("Training Random Forest model...")
    np.random.seed(42)
    n_samples = 2000
    
    study_hours = np.random.uniform(0, 12, n_samples)
    attendance = np.random.uniform(20, 100, n_samples)
    gpa = np.random.uniform(1.0, 4.0, n_samples)
    sleep = np.random.uniform(3, 10, n_samples)
    extra = np.random.randint(0, 2, n_samples)
    
    # Clear logical rule: Passes if study hours >= 3.5, attendance >= 65%, and gpa >= 2.2
    pass_condition = (study_hours >= 3.5) & (attendance >= 65.0) & (gpa >= 2.2)
    y = pass_condition.astype(int)
    
    X = np.column_stack((study_hours, attendance / 100.0, gpa / 4.0, sleep, extra))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, None

def predict_performance(model, scaler, input_data):
    prediction_proba = model.predict_proba(input_data)
    return float(prediction_proba[0][1])

def generate_recommendations(study_hours, attendance, prev_gpa, sleep_hours, probability):
    recommendations = []
    
    # If the model predicts a FAIL (< 0.5 probability), prioritize critical warnings
    if probability < 0.5:
        recommendations.append("🚨 **Critical Alert:** Your overall performance profile indicates a high risk of failure.")
        if study_hours < 3.5:
            recommendations.append(f"⚠️ **Study Hours:** Increase your daily study time from {study_hours:.1f}h to at least 4.0h.")
        if attendance < 65.0:
            recommendations.append(f"⚠️ **Attendance:** Your attendance is at {attendance:.1f}%. It must be above 65% to pass.")
        if prev_gpa < 2.2:
            recommendations.append(f"⚠️ **Academic Standing:** Your CGPA ({prev_gpa:.2f}) is below the safe threshold of 2.2.")
        if sleep_hours < 6.0:
            recommendations.append("⚠️ **Health:** Ensure you get at least 6-7 hours of sleep to maintain focus.")
            
        if len(recommendations) == 1: # Fallback if specific metrics look okay but combined logic failed
            recommendations.append("⚠️ **General Improvement:** Your combined metrics are borderline. Try boosting both study hours and attendance.")
    else:
        # Check individual warnings even for a pass if any metric is slightly low
        if study_hours < 4.0:
            recommendations.append("⚠️ **Study Hours Tip:** Consider increasing study time slightly for better grades.")
        if attendance < 75.0:
            recommendations.append("⚠️ **Attendance Tip:** Aim for above 75% attendance for a safer academic standing.")
        if prev_gpa < 2.5:
            recommendations.append("⚠️ **Academic Tip:** Focus on targeted revision to improve your CGPA.")
        if sleep_hours < 6.0:
            recommendations.append("⚠️ **Health Tip:** Insufficient sleep can impact cognitive performance.")
            
        if not recommendations:
            recommendations.append("🎉 **Excellent Status:** All metrics look great! Keep maintaining your current study habits.")
            
    return recommendations