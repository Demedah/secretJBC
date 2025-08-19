# (UI Streamlit)
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from uji_citra_core import train_model
from uji_citra_predict import predict_image
import os

st.title("🔬 Klasifikasi Jenis Kulit Wajah - Jiabao Clinic")

MODEL_PATH = "model_jbc.pkl"

# Training jika model belum ada
if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model dari database..."):
        model, le, acc = train_model(save_path=MODEL_PATH)
    st.success(f"✅ Model dilatih, akurasi: {acc:.2f}")
else:
    st.info("📦 Model sudah tersedia, langsung dipakai.")

# Upload gambar
uploaded_file = st.file_uploader("Upload Foto Wajah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    pred_label, classes, probs = predict_image(img, MODEL_PATH)

    st.success(f"🎯 Prediksi Jenis Kulit: **{pred_label}**")

    prob_df = pd.DataFrame({
        "Kelas": classes,
        "Probabilitas": probs
    }).sort_values("Probabilitas", ascending=False)

    st.dataframe(prob_df)
    st.bar_chart(prob_df.set_index("Kelas"))
