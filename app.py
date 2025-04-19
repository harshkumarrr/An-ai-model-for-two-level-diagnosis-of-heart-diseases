import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# Import your existing functions
from ecg_prediction_model import train_models, predict_from_image  # Replace 'your_module_name' with the filename (without .py)

# --- Streamlit GUI ---
st.set_page_config(page_title="ECG Image-Based Heart Disease Diagnosis", layout="centered")

st.title("🫀 ECG Image-Based Heart Disease Diagnosis")
st.markdown("Upload an ECG image to analyze potential heart diseases using AI.")

uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG", use_column_width=True)

    # Save temporary image to pass into your function
    temp_image_path = "temp_ecg_image.jpg"
    image.save(temp_image_path)

    with st.spinner("Analyzing ECG..."):
        # Train/load models (do this once per session ideally)
        if "rf_model" not in st.session_state:
            st.session_state.rf_model, st.session_state.xgb_model = train_models()

        result = predict_from_image(
            temp_image_path,
            st.session_state.rf_model,
            st.session_state.xgb_model
        )

    # Display Results
    st.subheader("📋 Diagnosis Result")
    st.markdown(f"**Diagnosis**: `{result['Diagnosis']}`")
    st.markdown(f"**Specific Disease**: `{result['Specific Disease']}`")
    st.markdown(f"**Confidence Score**: `{result['Confidence Score'] * 100:.1f}%`")

    if result["Factors"]:
        st.subheader("🔍 Detected Anomalies")
        for k, v in result["Factors"].items():
            st.markdown(f"- **{k}**: {v}")
    else:
        st.success("All features within normal range.")

    st.markdown("---")
    st.caption("Model trained on synthetic ECG data. For demonstration only.")

