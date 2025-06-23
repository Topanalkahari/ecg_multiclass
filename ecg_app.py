import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from ecg_pipeline import ECGPipeline

# Title of apps
st.set_page_config(page_title="ECG Prediction App", layout="centered")
st.title("📈 ECG Prediction System")

# pipeline
pipeline = ECGPipeline()

# Load model dan scaler
model_path = "ecg_model.pkl"
scaler_path = "ecg_scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    pipeline.predictor.model = joblib.load(model_path)
    pipeline.predictor.scaler = joblib.load(scaler_path)
    pipeline.predictor.is_trained = True
    st.success("✅ Model loaded successfully!")
else:
    st.warning("⚠️ Model belum tersedia. Harap latih dan simpan model terlebih dahulu.")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("📤 Upload gambar ECG (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # temp save
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())
    
    st.image("temp_image.jpg", caption="Gambar ECG yang diunggah", use_column_width=True)

    # process and preds
    with st.spinner("🔎 Memproses gambar dan memprediksi..."):
        try:
            predictions, probabilities = pipeline.predict_new_image("temp_image.jpg")
            csv_data, _ = pipeline.process_single_image("temp_image.jpg", save_csv=False)

            # Tampilkan prediksi
            st.subheader("📊 Hasil Prediksi")
            st.write(f"**Prediksi:** {predictions[0]}")
            st.write(f"**Confidence:** {np.max(probabilities[0]):.4f}")

            # Tampilkan sinyal ECG
            st.subheader("📉 Visualisasi Sinyal ECG")
            fig, ax = plt.subplots()
            ax.plot(csv_data["time"], csv_data["ecg_signal"], color='blue')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized ECG Signal")
            ax.set_title("Sinyal ECG Terproses")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error saat memproses gambar: {e}")
else:
    st.info("👈 Silakan upload gambar ECG terlebih dahulu.")