# uji_citra.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============ 1) Fungsi Ekstraksi Fitur ============
def ekstrak_fitur_gambar(img_input):
    try:
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                return None
        elif isinstance(img_input, np.ndarray):
            img = img_input
        else:
            return None

        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Fitur GLCM
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # Histogram Warna
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()

        hist_b = hist_b / hist_b.sum() if hist_b.sum() != 0 else hist_b
        hist_g = hist_g / hist_g.sum() if hist_g.sum() != 0 else hist_g
        hist_r = hist_r / hist_r.sum() if hist_r.sum() != 0 else hist_r

        fitur_hist = np.hstack([hist_b, hist_g, hist_r])
        fitur = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, fitur_hist])
        return fitur
    except:
        return None

# ============ 2) Load Dataset dari GitHub ============
image_dir = "Extraksi"
df = pd.read_csv("databaseJBC.csv")

import ast
import numpy as np

X, y = [], []
for idx, row in df.iterrows():
    try:
        fitur = ast.literal_eval(row["pixel_features"])
        fitur = [float(v) for v in fitur]   # pastikan float
        X.append(fitur)
        y.append(row["Tekstur Kulit"])
    except Exception as e:
        print(f"Row {idx} error: {e}")

X = np.array(X, dtype=float)
y = np.array(y)
print("Final shape X:", X.shape)


# 3. Encode label y
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Split dataset (kalau cukup besar)
if len(X) >= 5:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    print("⚠️ Dataset kecil, semua dipakai training.")
    X_train, y_train = X, y
    X_test, y_test = X, y

# 5. Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi
if len(X_test) > 0:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Akurasi:", acc)
    
# ============ 3) Streamlit UI ============
st.title("Klasifikasi Jenis Kulit Wajah - Jiabao Clinic")
st.write(f"Akurasi Model: **{acc:.2f}**")

uploaded_file = st.file_uploader("Upload Foto Wajah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Gambar diupload", use_column_width=True)

    # Ekstrak fitur dari gambar
    fitur = ekstrak_fitur_gambar(img)
    if fitur is not None:
        expected_len = model.n_features_in_

        # Debug info
        st.sidebar.write(f"📊 Fitur model: {expected_len}")
        st.sidebar.write(f"📊 Fitur input: {len(fitur)}")

        # Jika fitur terlalu panjang -> potong
        if len(fitur) > expected_len:
            fitur = fitur[:expected_len]
        # Jika fitur terlalu pendek -> padding dengan nol
        elif len(fitur) < expected_len:
            fitur = list(fitur) + [0.0] * (expected_len - len(fitur))

        # Prediksi dengan model
        pred = model.predict([fitur])[0]
        probs = model.predict_proba([fitur])[0]

        # Tampilkan hasil prediksi
        st.success(f"Prediksi Jenis Kulit: **{pred}**")

        # Tampilkan probabilitas
        prob_df = pd.DataFrame({
            "Kelas": le.classes_,
            "Probabilitas": probs
        }).sort_values("Probabilitas", ascending=False)

        st.dataframe(prob_df)
        st.bar_chart(prob_df.set_index("Kelas"))

    else:
        st.error("⚠️ Gagal ekstrak fitur dari gambar yang diupload.")

