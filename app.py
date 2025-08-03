import streamlit as st
import numpy as np
import cv2
import joblib
from extract_features import extract_all_features

model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Parkinson's Disease Detection 🧠")
st.markdown("Upload a **spiral/wave drawing image** to check for Parkinson’s.")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Select Gender", "Male", "Female", "Other"])
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    submit = st.form_submit_button("Predict")

if submit:
    if not name or gender == "Select Gender" or uploaded is None:
        st.warning("⚠️ Please fill all patient details and upload an image.")
    else:
        try:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite("temp_image.jpg", image)

            features = extract_all_features("temp_image.jpg")
            scaled = scaler.transform([features])
            prediction = model.predict(scaled)[0]

            st.subheader(f"👤 Patient: {name}, {age} y/o, {gender}")
            if prediction == 0:
                st.success("🟢 Healthy")
            else:
                st.error("🔴 Parkinson's Detected")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")