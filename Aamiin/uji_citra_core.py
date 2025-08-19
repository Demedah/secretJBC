# dari database
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
import joblib   # untuk save/load model

def ekstrak_fitur_gambar(img):
    """Ekstraksi fitur GLCM + histogram warna"""
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Histogram warna
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()

    hist_b = hist_b / hist_b.sum() if hist_b.sum() != 0 else hist_b
    hist_g = hist_g / hist_g.sum() if hist_g.sum() != 0 else hist_g
    hist_r = hist_r / hist_r.sum() if hist_r.sum() != 0 else hist_r

    fitur_hist = np.hstack([hist_b, hist_g, hist_r])
    fitur = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, fitur_hist])
    return fitur

def train_model(csv_file="databaseJBC.csv", image_dir="Extraksi", save_path="model_jbc.pkl"):
    """Training model dari database CSV + gambar"""
    if not os.path.exists(csv_file):
        print("❌ Dataset tidak ditemukan")
        return None, None, 0.0

    df = pd.read_csv(csv_file)
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest"
    )

    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["FotoCS"])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        fitur_asli = ekstrak_fitur_gambar(img)
        X.append(fitur_asli)
        y.append(row["Tekstur Kulit"])

        # Augmentasi 3x
        img_expanded = np.expand_dims(img, axis=0)
        for i in range(3):
            aug_img = datagen.flow(img_expanded, batch_size=1)[0][0].astype("uint8")
            fitur_aug = ekstrak_fitur_gambar(aug_img)
            X.append(fitur_aug)
            y.append(row["Tekstur Kulit"])

    X = np.array(X, dtype=float)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # simpan model + label encoder
    joblib.dump((model, le), save_path)
    print(f"✅ Model disimpan di {save_path}, akurasi: {acc:.3f}")
    return model, le, acc
