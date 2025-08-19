# dari database
# train_model.py
import pandas as pd
import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.feature import graycomatrix, graycoprops

# Fungsi ekstraksi fitur
def ekstrak_fitur_gambar(img):
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]

    # Histogram warna
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        hist /= hist.sum() if hist.sum() != 0 else 1
        hist_features.extend(hist)

    return np.hstack([features, hist_features])

# Load dataset
df = pd.read_csv("databaseJBC.csv")
image_dir = "Extraksi"

X, y = [], []
for _, row in df.iterrows():
    img_path = os.path.join(image_dir, row["FotoCS"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    fitur = ekstrak_fitur_gambar(img)
    X.append(fitur)
    y.append(row["Tekstur Kulit"])

X, y = np.array(X), np.array(y)

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
acc = accuracy_score(y_test, model.predict(X_test))
print("Akurasi:", acc)

# Simpan model dan encoder
joblib.dump((model, le), "model_jbc.pkl")
print("✅ Model saved to model_jbc.pkl")
