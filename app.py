import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="EduPredict", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    .main, .block-container {
        background-color: #ffffff !important;
        max-width: 1000px;
        padding-top: 2.5rem;
    }
    h1 {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #111827 !important;
        text-align: center;
        margin-bottom: 4px !important;
    }
    h3 {
        font-size: 16px !important;
        text-align: center;
        color: #6b7280 !important;
        font-weight: 400 !important;
        margin-top: -8px !important;
        margin-bottom: 4px !important;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-size: 11px;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }
    .stSlider label, .stSelectbox label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-color: #e5e7eb !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        width: 100%;
        background-color: #111827 !important;
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 14px 0 !important;
        border-radius: 10px !important;
        border: none !important;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #1f2937 !important;
    }
    .result-box {
        border-radius: 12px;
        padding: 28px 20px;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-pass {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
    }
    .result-fail {
        background-color: #fff1f2;
        border: 1px solid #fda4af;
    }
    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .result-confidence {
        font-size: 13px;
        color: #6b7280;
    }
    .section-title {
        font-size: 11px;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('model/student_model.pkl')
le_gender = joblib.load('model/le_gender.pkl')
le_parent = joblib.load('model/le_parent.pkl')
le_internet = joblib.load('model/le_internet.pkl')
le_extra = joblib.load('model/le_extra.pkl')
le_result = joblib.load('model/le_result.pkl')

st.markdown("<h1>EduPredict</h1>", unsafe_allow_html=True)
st.markdown("<h3>Student Performance Predictor</h3>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter student details below to predict academic outcome</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Personal Information</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.slider("Age", 15, 20, 17)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col3:
    parent_education = st.selectbox("Parent Education Level", ["Primary", "Secondary", "Graduate", "Postgraduate"])
with col4:
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Academic Details</div>', unsafe_allow_html=True)
col5, col6, col7, col8 = st.columns(4)
with col5:
    study_hours = st.slider("Study Hours per Day", 1.0, 10.0, 5.0)
with col6:
    attendance = st.slider("Attendance (%)", 50.0, 100.0, 75.0)
with col7:
    previous_marks = st.slider("Previous Marks", 40, 100, 65)
with col8:
    extra_classes = st.selectbox("Extra Classes", ["Yes", "No"])
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict Performance"):
    input_data = pd.DataFrame({
        'age': [age],
        'gender': le_gender.transform([gender]),
        'study_hours_per_day': [study_hours],
        'attendance_percent': [attendance],
        'previous_marks': [previous_marks],
        'parent_education': le_parent.transform([parent_education]),
        'internet_access': le_internet.transform([internet_access]),
        'extra_classes': le_extra.transform([extra_classes]),
    })

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]
    result = le_result.inverse_transform(prediction)[0]
    confidence = round(max(proba) * 100, 1)

    if result == "Pass":
        st.markdown(f"""
            <div class="result-box result-pass">
                <div class="result-title" style="color:#15803d;">Pass</div>
                <div class="result-confidence">Model Confidence: {confidence}%</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box result-fail">
                <div class="result-title" style="color:#be123c;">Fail</div>
                <div class="result-confidence">Model Confidence: {confidence}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Student Report Card</div>', unsafe_allow_html=True)

    suggestions = []
    if study_hours < 4:
        suggestions.append("Study hours kam hain — rozana kam se kam 4-5 ghante padhne ki koshish karo.")
    if attendance < 75:
        suggestions.append("Attendance 75% se kam hai — class attendance improve karo.")
    if previous_marks < 50:
        suggestions.append("Previous marks kam hain — weak subjects pe zyada focus karo.")
    if extra_classes == "No":
        suggestions.append("Extra classes join karo — doubt clear karne mein madad milegi.")
    if internet_access == "No":
        suggestions.append("Internet access nahi hai — library ya study center use karo online resources ke liye.")
    if not suggestions:
        suggestions.append("Sab kuch achha lag raha hai — aise hi mehnat karte raho!")

    report_data = {
        "Parameter": ["Study Hours/Day", "Attendance", "Previous Marks", "Extra Classes", "Internet Access", "Predicted Result"],
        "Value": [f"{study_hours} hrs", f"{attendance}%", f"{previous_marks}/100", extra_classes, internet_access, result],
        "Status": [
            "Good" if study_hours >= 4 else "Low",
            "Good" if attendance >= 75 else "Low",
            "Good" if previous_marks >= 50 else "Low",
            "Good" if extra_classes == "Yes" else "No",
            "Good" if internet_access == "Yes" else "No",
            "Pass" if result == "Pass" else "Fail"
        ]
    }

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title" style="margin-top:1rem;">Suggestions</div>', unsafe_allow_html=True)
    for s in suggestions:
        st.markdown(f"<p style='font-size:13px; color:#374151; padding:8px 12px; background:#f9fafb; border-radius:8px; margin-bottom:6px; border-left:3px solid #111827;'>{s}</p>", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    features = ['Age', 'Gender', 'Study Hours', 'Attendance',
                'Previous Marks', 'Parent Education', 'Internet', 'Extra Classes']
    importance = model.feature_importances_
    indices = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#f9fafb')
    ax.set_facecolor('#f9fafb')
    ax.barh([features[i] for i in indices], importance[indices], color='#111827', height=0.5)
    ax.set_xlabel('Importance Score', fontsize=11, color='#6b7280')
    ax.tick_params(colors='#374151', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    plt.tight_layout()
    st.pyplot(fig)