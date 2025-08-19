# uji_citra_web.py
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
import plotly.express as px
import plotly.graph_objects as go

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

        # Fitur GLCM
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # Histogram Warna
        hist_b = cv2.calcHist([img], [0], None, [64], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [64], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [64], [0, 256]).flatten()

        hist_b = hist_b / (hist_b.sum() + 1e-6)
        hist_g = hist_g / (hist_g.sum() + 1e-6)
        hist_r = hist_r / (hist_r.sum() + 1e-6)

        fitur_hist = np.hstack([hist_b, hist_g, hist_r])
        fitur = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, fitur_hist])
        return fitur
    except:
        return None

# ================== STREAMLIT APP ==================
st.set_page_config(page_title="Klasifikasi Kulit Jiabao", page_icon="🧴", layout="wide")

st.title("🧴 Klasifikasi Jenis Kulit Wajah - Jiabao Clinic")
st.markdown("Aplikasi ini menggunakan **Random Forest** + **GLCM & Histogram Warna**")

# ================== LOAD DATASET ==================
if not os.path.exists("databaseJBC.csv"):
    st.error("❌ Dataset `databaseJBC.csv` tidak ditemukan! Pastikan file tersedia.")
    st.stop()

df = pd.read_csv("databaseJBC.csv")
st.success(f"✅ Dataset berhasil dimuat: {len(df)} sampel")

# ================== TRAIN MODEL ==================
X, y = [], []
for idx, row in df.iterrows():
    img_path = os.path.join("Extraksi", row["FotoCS"])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
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

# ================== DASHBOARD ==================
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

st.subheader("Distribusi Jenis Kulit")
fig_dist = px.histogram(df, x="Tekstur Kulit", title="Distribusi Kelas")
st.plotly_chart(fig_dist, use_container_width=True)

# ================== TES GAMBAR ==================
st.markdown("---")
st.header("🔍 Tes Prediksi Gambar")

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
        fig_prob = px.bar(prob_df, x="Probability", y="Skin Type", orientation='h',
                          title="Confidence Prediction", color="Probability", color_continuous_scale="viridis")
        st.plotly_chart(fig_prob, use_container_width=True)
        
        skin_info = {
            "dry": "Kulit kering: cenderung kasar, mudah pecah-pecah, butuh pelembap ekstra.",
            "oily": "Kulit berminyak: mudah berjerawat, perlu pembersih yang mengontrol minyak.",
            "normal": "Kulit normal: seimbang, tidak terlalu kering/berminyak.",
            "combination": "Kulit kombinasi: berminyak di T-zone, kering di area lain."
        }
        st.info(skin_info.get(pred_label.lower(), "Informasi tidak tersedia untuk jenis kulit ini."))

        # ============ Tambahan rekomendasi ============
        label_lower = pred_label.lower()

        if "oily" in label_lower or "berminyak" in label_lower:
            st.markdown("""
            ### 💡 Rekomendasi Treatment untuk Kulit Berminyak
            - ✨ **Chemical Peeling (AHA/BHA peeling)** → mengurangi minyak & membersihkan pori.  
            - 💎 **Laser karbon / carbon peel** → mengontrol sebum & mengecilkan pori.  
            - 💉 **Microneedling + serum niacinamide** → mengurangi bekas jerawat & produksi minyak.  
            - 🔆 **IPL (Intense Pulsed Light)** → membantu atasi jerawat aktif.  
            - 🧖‍♀️ **Deep cleansing facial** khusus oily skin (angkat komedo & sebum berlebih).  
            """)
        elif "dry" in label_lower or "kering" in label_lower:
            st.markdown("""
            ### 💡 Rekomendasi Treatment untuk Kulit Kering
            - 💧 **HydraFacial** → pembersihan + infus serum hydrating.  
            - 💉 **Infus vitamin / skin booster (HA, collagen)** → melembapkan dari dalam.  
            - 🌬️ **Oxy facial / Oxy infusion** → memberi oksigen & serum kelembapan.  
            - 💎 **Mesotherapy (HA, peptide)** → suntikan microdose untuk hidrasi kulit.  
            - 🔴 **LED therapy (red light)** → memperbaiki barrier & merangsang kolagen.  
            """)
        elif "normal" in label_lower:
            st.markdown("""
            ### 💡 Rekomendasi Treatment untuk Kulit Normal
            - 🌸 **Facial rutin** (brightening facial, hydrafacial).  
            - ✨ **Mild chemical peel** → regenerasi sel kulit.  
            - 💡 **Laser toning / Rejuvenation** → menjaga kecerahan.  
            - 💉 **Microneedling ringan** → anti-aging & elastisitas.  
            - 🩸 **PRP (Platelet Rich Plasma)** → peremajaan kulit jangka panjang.  
            """)
        else:
            st.warning("⚠️ Tidak ada rekomendasi khusus untuk label ini.")

