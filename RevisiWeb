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
from plotly.subplots import make_subplots
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
        return fitur, (hist_b, hist_g, hist_r), (contrast, dissimilarity, homogeneity, energy, correlation)
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
    result = ekstrak_fitur_gambar(img)
    if result is not None:
        fitur_asli = result[0] if isinstance(result, tuple) else result
        X.append(fitur_asli)
        y.append(label)

    # augmentasi (5 variasi per gambar)
    img_expanded = np.expand_dims(img, axis=0)
    aug_iter = datagen.flow(img_expanded, batch_size=1)
    for i in range(5):
        aug_img = next(aug_iter)[0].astype("uint8")
        result = ekstrak_fitur_gambar(aug_img)
        if result is not None:
            fitur_aug = result[0] if isinstance(result, tuple) else result
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


# ================== DASHBOARD ==================
st.header("🔬 Website Jiabao Clinic")
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

    result = ekstrak_fitur_gambar(img)
    if result is not None:
        if isinstance(result, tuple):
            fitur, (hist_b, hist_g, hist_r), (contrast, dissimilarity, homogeneity, energy, correlation) = result
        else:
            fitur = result
            hist_b = hist_g = hist_r = None
            contrast = dissimilarity = homogeneity = energy = correlation = 0
        
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
                          title="Confidence Prediction", color="Probability", 
                          color_continuous_scale="viridis",
                          text="Probability")
        fig_prob.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_prob.update_layout(height=400)
        st.plotly_chart(fig_prob, use_container_width=True)

        st.subheader("📊 Analisis Detail Gambar")
        
        # Color Histogram Analysis
        if hist_b is not None and hist_g is not None and hist_r is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # RGB Histogram
                fig_hist = go.Figure()
                x_values = list(range(256))
                fig_hist.add_trace(go.Scatter(x=x_values, y=hist_r, mode='lines', name='Red', line=dict(color='red')))
                fig_hist.add_trace(go.Scatter(x=x_values, y=hist_g, mode='lines', name='Green', line=dict(color='green')))
                fig_hist.add_trace(go.Scatter(x=x_values, y=hist_b, mode='lines', name='Blue', line=dict(color='blue')))
                fig_hist.update_layout(
                    title="Distribusi Histogram Warna RGB",
                    xaxis_title="Intensitas Pixel",
                    yaxis_title="Frekuensi Normalisasi",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Color dominance pie chart
                avg_r = np.mean(hist_r)
                avg_g = np.mean(hist_g) 
                avg_b = np.mean(hist_b)
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Red', 'Green', 'Blue'],
                    values=[avg_r, avg_g, avg_b],
                    marker_colors=['red', 'green', 'blue']
                )])
                fig_pie.update_layout(
                    title="Dominasi Warna Rata-rata",
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # GLCM Features Radar Chart
        glcm_features = {
            'Contrast': contrast,
            'Dissimilarity': dissimilarity, 
            'Homogeneity': homogeneity,
            'Energy': energy,
            'Correlation': correlation
        }
        
        # Normalize GLCM values for better radar chart visualization
        max_val = max(glcm_features.values())
        normalized_glcm = {k: (v/max_val)*100 for k, v in glcm_features.items()}
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(normalized_glcm.values()),
            theta=list(normalized_glcm.keys()),
            fill='toself',
            name='Fitur GLCM',
            line_color='rgb(1,90,180)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Profil Tekstur Kulit (GLCM Features)",
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # GLCM Features Bar Chart with actual values
        fig_glcm_bar = px.bar(
            x=list(glcm_features.keys()),
            y=list(glcm_features.values()),
            title="Nilai Fitur GLCM",
            labels={'x': 'Fitur GLCM', 'y': 'Nilai'},
            color=list(glcm_features.values()),
            color_continuous_scale='viridis'
        )
        fig_glcm_bar.update_layout(height=400)
        st.plotly_chart(fig_glcm_bar, use_container_width=True)

        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'] + \
                          [f'Hist_B_{i}' for i in range(256)] + \
                          [f'Hist_G_{i}' for i in range(256)] + \
                          [f'Hist_R_{i}' for i in range(256)]
            
            # Get top 20 most important features
            top_indices = np.argsort(model.feature_importances_)[-20:]
            top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in top_indices]
            top_importances = model.feature_importances_[top_indices]
            
            fig_importance = px.bar(
                x=top_importances,
                y=top_features,
                orientation='h',
                title="Top 20 Fitur Paling Penting untuk Prediksi",
                labels={'x': 'Importance Score', 'y': 'Features'},
                color=top_importances,
                color_continuous_scale='plasma'
            )
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)
        
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
