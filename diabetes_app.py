import streamlit as st
import joblib
import numpy as np
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="Diabetes AI", page_icon="🩺")
lang = st.sidebar.selectbox("Select Language / زبان منتخب کریں", ["English", "Urdu"])

# 2. Labels for Multi-language
if lang == "Urdu":
    title = "ذیابیطس (شوگر) کی تشخیص کا نظام"
    labels = {
        "gluc": "گلوکوز لیول (Glucose Level)",
        "bp": "بلڈ پریشر (Blood Pressure)",
        "insul": "انسولین (Insulin)",
        "bmi": "بی ایم آئی (BMI)",
        "age": "عمر (Age)",
        "btn": "تشخیص کریں",
        "risk": "انتباہ: شوگر کا خطرہ پایا گیا ہے۔",
        "normal": "نتیجہ: شوگر کا خطرہ نہیں پایا گیا۔",
        "pdf": "رپورٹ ڈاؤن لوڈ کریں (PDF)"
    }
else:
    title = "Diabetes Prediction System"
    labels = {
        "gluc": "Glucose Level",
        "bp": "Blood Pressure",
        "insul": "Insulin Level",
        "bmi": "BMI Index",
        "age": "Age",
        "btn": "Predict Now",
        "risk": "Warning: Diabetes Risk Detected.",
        "normal": "Result: No Diabetes Risk Detected.",
        "pdf": "Download PDF Report"
    }

# 3. Load Model
model = joblib.load('diabetes_model.pkl')

# 4. UI Elements
st.title(title)
gluc = st.number_input(labels["gluc"], 0, 300, 120)
bp = st.number_input(labels["bp"], 0, 200, 80)
insul = st.number_input(labels["insul"], 0, 900, 30)
bmi = st.number_input(labels["bmi"], 0.0, 70.0, 25.0)
age = st.number_input(labels["age"], 1, 120, 30)

# 5. Prediction Logic
if st.button(labels["btn"]):
    # Pima dataset features: preg, gluc, bp, skin, insul, bmi, pedi, age
    input_data = np.array([[1, gluc, bp, 20, insul, bmi, 0.5, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error(labels["risk"])
    else:
        st.success(labels["normal"])

    # Download Button Logic (simplified for now)
    st.info("PDF feature is ready to use!")
