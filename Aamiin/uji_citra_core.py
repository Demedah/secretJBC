# uji_citra_core.py
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

# Fungsi ekstraksi fitur
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


# Fungsi training model
def train_model(csv_file="databaseJBC.csv", image_dir="Extraksi"):
    if not os.path.exists(csv_file):
        print("❌ Dataset CSV tidak ditemukan!")
        return None, None, 0.0

    df = pd.read_csv(csv_file)
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

    for _, row in df.iterrows():
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

        # augmentasi
        img_expanded = np.expand_dims(img, axis=0)
        aug_iter = datagen.flow(img_expanded, batch_size=1)
        for _ in range(3):  # 3 augmentasi per gambar
            aug_img = next(aug_iter)[0].astype("uint8")
            fitur_aug = ekstrak_fitur_gambar(aug_img)
            if fitur_aug is not None:
                X.append(fitur_aug)
                y.append(label)

    X = np.array(X, dtype=float)
    y = np.array(y)

    if len(X) < 5:
        print("❌ Data terlalu sedikit untuk training!")
        return None, None, 0.0

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, le, acc
