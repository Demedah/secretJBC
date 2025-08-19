# untuk upload
import cv2
import numpy as np
import joblib
from uji_citra_core import ekstrak_fitur_gambar

def predict_image(img, model_path="model_jbc.pkl"):
    """Prediksi jenis kulit dari gambar upload"""
    model, le = joblib.load(model_path)
    fitur = ekstrak_fitur_gambar(img)

    expected_len = model.n_features_in_
    if len(fitur) > expected_len:
        fitur = fitur[:expected_len]
    elif len(fitur) < expected_len:
        fitur = list(fitur) + [0.0] * (expected_len - len(fitur))

    pred_encoded = model.predict([fitur])[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba([fitur])[0]
    return pred_label, le.classes_, probs
