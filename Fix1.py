# uji_citra_web.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.express as px
import plotly.graph_objects as go

# ================== CUSTOM CSS ==================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #f9f9f9, #eef2f3);
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #20c997;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== EKSTRAKSI FITUR ==================
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
    except:
        return None

# ================== LOAD DATASET ==================
st.title("🔬 Klasifikasi Jenis Kulit Wajah - Jiabao Clinic")
st.markdown("---")

if not os.path.exists("databaseJBC.csv"):
    st.error("❌ Dataset 'databaseJBC.csv' tidak ditemukan!")
    st.stop()

df = pd.read_csv("databaseJBC.csv")
st.success(f"✅ Dataset berhasil dimuat: {len(df)} sampel")

image_dir = "Extraksi"
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ================== TRAIN MODEL ==================
X, y = [], []
for idx, row in df.iterrows():
    img_path = os.path.join(image_dir, row["FotoCS"])
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue

    fitur = ekstrak_fitur_gambar(img)
    if fitur is not None:
        X.append(fitur)
        y.append(row["Tekstur Kulit"])

X = np.array(X, dtype=float)
y = np.array(y)

if len(X) < 5:
    st.error("❌ Data tidak cukup untuk melatih model.")
    st.stop()

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ================== DASHBOARD LAYOUT ==================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Tes Gambar", "ℹ️ Tentang"])

# ---------- TAB 1: DASHBOARD ----------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Data", f"{len(df)} sampel")
        st.metric("Jumlah Kelas", f"{len(le.classes_)} jenis kulit")
    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=acc*100,
            title={'text': "Akurasi Model (%)"},
            gauge={'axis': {'range': [0,100]},
                   'bar': {'color': "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightcoral"},
                       {'range': [50, 80], 'color': "gold"},
                       {'range': [80, 100], 'color': "lightgreen"}]}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Distribusi Kelas Kulit")
    fig_dist = px.histogram(df, x="Tekstur Kulit", title="Distribusi Jenis Kulit")
    st.plotly_chart(fig_dist, use_container_width=True)

# ---------- TAB 2: TES GAMBAR ----------
with tab2:
    uploaded_file = st.file_uploader("Upload Foto Wajah", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        fitur = ekstrak_fitur_gambar(img)
        if fitur is not None:
            if len(fitur) < model.n_features_in_:
                fitur = list(fitur) + [0.0]*(model.n_features_in_-len(fitur))
            elif len(fitur) > model.n_features_in_:
                fitur = fitur[:model.n_features_in_]

            pred_encoded = model.predict([fitur])[0]
            pred_label = le.inverse_transform([pred_encoded])[0]
            probs = model.predict_proba([fitur])[0]

            st.success(f"🎯 Prediksi Jenis Kulit: **{pred_label}**")

            prob_df = pd.DataFrame({"Skin Type": le.classes_, "Probability": probs})
            fig_prob = px.bar(prob_df, x="Probability", y="Skin Type", orientation='h', color="Probability",
                              title="Confidence Prediction", color_continuous_scale="viridis")
            st.plotly_chart(fig_prob, use_container_width=True)

            # Info tambahan jenis kulit
            skin_info = {
                "dry": "Kulit kering: cenderung kasar, mudah pecah-pecah, butuh pelembap ekstra.",
                "oily": "Kulit berminyak: mudah berjerawat, perlu pembersih yang mengontrol minyak.",
                "normal": "Kulit normal: seimbang, tidak terlalu kering/berminyak.",
                "combination": "Kulit kombinasi: berminyak di T-zone (dahi, hidung, dagu), kering di area lain."
            }
            st.info(skin_info.get(pred_label.lower(), "Informasi tidak tersedia untuk jenis kulit ini."))

# ---------- TAB 3: TENTANG ----------
with tab3:
    st.markdown("""
    ### ℹ️ Tentang Aplikasi
    - **Feature Extraction**: GLCM (Gray-Level Co-occurrence Matrix) + Color Histogram  
    - **Classification**: Random Forest (100 trees)  
    - **Data Augmentation**: Rotasi, geser, zoom, flip  
    - **Visualization**: Plotly, Gauge Chart, Histogram  

    **Tujuan:**  
    Membantu **klinik Jiabao** menganalisis jenis kulit wajah berdasarkan citra, 
    sehingga rekomendasi perawatan bisa lebih tepat sasaran.
    """)
