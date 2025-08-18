# uji_citra.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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


# ============ 2) Load Dataset & Augmentasi ============
df = pd.read_csv("databaseJBC.csv")
image_dir = "Extraksi"   # folder tempat gambar disimpan

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

X, y = [], []

for idx, row in df.iterrows():
    file_name = row["FotoCS"]
    label = row["Tekstur Kulit"]   

    img_path = os.path.join(image_dir, file_name)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (128, 128))

    # fitur asli
    fitur_asli = ekstrak_fitur_gambar(img)
    if fitur_asli is not None:
        X.append(fitur_asli)
        y.append(label)

    # augmentasi (5 variasi per gambar)
    img_expanded = np.expand_dims(img, axis=0)
    aug_iter = datagen.flow(img_expanded, batch_size=1)
    for i in range(5):
        aug_img = next(aug_iter)[0].astype("uint8")
        fitur_aug = ekstrak_fitur_gambar(aug_img)
        if fitur_aug is not None:
            X.append(fitur_aug)
            y.append(label)

X = np.array(X, dtype=float)
y = np.array(y)
print("Jumlah data setelah augmentasi:", X.shape, len(y))

# ============ 3) Label Encoding + Training ============
if len(X) > 5:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
else:
    model, le = None, None
    acc = 0.0


# ============ 4) Streamlit UI ============
st.title("Klasifikasi Jenis Kulit Wajah - Jiabao Clinic")
st.write(f"Akurasi Model: **{acc:.2f}**")

uploaded_file = st.file_uploader("Upload Foto Wajah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

st.image(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    caption="Gambar diupload",
    use_container_width=True
)


    fitur = ekstrak_fitur_gambar(img)
    if fitur is not None:
        expected_len = model.n_features_in_

        if len(fitur) > expected_len:
            fitur = fitur[:expected_len]
        elif len(fitur) < expected_len:
            fitur = list(fitur) + [0.0] * (expected_len - len(fitur))

        pred_encoded = model.predict([fitur])[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        probs = model.predict_proba([fitur])[0]

        st.success(f"Prediksi Jenis Kulit: **{pred_label}**")

        prob_df = pd.DataFrame({
            "Kelas": le.classes_,
            "Probabilitas": probs
        }).sort_values("Probabilitas", ascending=False)

        st.dataframe(prob_df)
        st.bar_chart(prob_df.set_index("Kelas"))

    else:
        st.error("⚠️ Gagal ekstrak fitur dari gambar yang diupload.")
