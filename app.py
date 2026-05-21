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
        suggestions.append("Study hours are low — try to study at least 4-5 hours daily.")
    if attendance < 75:
        suggestions.append("Attendance is below 75% — improve your class attendance.")
    if previous_marks < 50:
        suggestions.append("Previous marks are low — focus more on weak subjects.")
    if extra_classes == "No":
        suggestions.append("Join extra classes — it will help in clearing doubts.")
    if internet_access == "No":
        suggestions.append("No internet access — use library or study center for online resources.")
    if not suggestions:
        suggestions.append("Everything looks good — keep up the hard work!")

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

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)
st.markdown("<p style='font-size:13px; color:#6b7280; margin-bottom:1rem;'>Upload a CSV file — predict results for all students at once.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.markdown('<div class="section-title">Uploaded Data</div>', unsafe_allow_html=True)
    st.dataframe(batch_df, use_container_width=True, hide_index=True)

    try:
        batch_df['gender'] = le_gender.transform(batch_df['gender'])
        batch_df['parent_education'] = le_parent.transform(batch_df['parent_education'])
        batch_df['internet_access'] = le_internet.transform(batch_df['internet_access'])
        batch_df['extra_classes'] = le_extra.transform(batch_df['extra_classes'])

        X_batch = batch_df.drop(['student_id', 'final_marks', 'result'], axis=1, errors='ignore')
        predictions = model.predict(X_batch)
        batch_df['Predicted Result'] = le_result.inverse_transform(predictions)

        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.dataframe(batch_df[['student_id', 'study_hours_per_day', 'attendance_percent', 'previous_marks', 'Predicted Result']], use_container_width=True, hide_index=True)

        pass_count = (batch_df['Predicted Result'] == 'Pass').sum()
        fail_count = (batch_df['Predicted Result'] == 'Fail').sum()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"<div style='background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:16px; text-align:center;'><div style='font-size:22px; font-weight:700; color:#15803d;'>{pass_count}</div><div style='font-size:13px; color:#6b7280;'>Students will Pass</div></div>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"<div style='background:#fff1f2; border:1px solid #fda4af; border-radius:10px; padding:16px; text-align:center;'><div style='font-size:22px; font-weight:700; color:#be123c;'>{fail_count}</div><div style='font-size:13px; color:#6b7280;'>Students might Fail</div></div>", unsafe_allow_html=True)

        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "predicted_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e} — CSV format is incorrect.")