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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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

def analisis_detail_gambar(img):
    """Fungsi untuk analisis detail GLCM dan RGB dari gambar yang diupload"""
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # GLCM Analysis
    glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], symmetric=True, normed=True)
    
    glcm_features = {}
    for angle_idx, angle in enumerate([0, 45, 90, 135]):
        glcm_features[f'contrast_{angle}'] = graycoprops(glcm, 'contrast')[0, angle_idx]
        glcm_features[f'dissimilarity_{angle}'] = graycoprops(glcm, 'dissimilarity')[0, angle_idx]
        glcm_features[f'homogeneity_{angle}'] = graycoprops(glcm, 'homogeneity')[0, angle_idx]
        glcm_features[f'energy_{angle}'] = graycoprops(glcm, 'energy')[0, angle_idx]
        glcm_features[f'correlation_{angle}'] = graycoprops(glcm, 'correlation')[0, angle_idx]
    
    # RGB Analysis
    b, g, r = cv2.split(img_resized)
    
    rgb_stats = {
        'red_mean': np.mean(r),
        'green_mean': np.mean(g),
        'blue_mean': np.mean(b),
        'red_std': np.std(r),
        'green_std': np.std(g),
        'blue_std': np.std(b),
        'red_max': np.max(r),
        'green_max': np.max(g),
        'blue_max': np.max(b),
        'red_min': np.min(r),
        'green_min': np.min(g),
        'blue_min': np.min(b)
    }
    
    # Histogram data
    hist_r = cv2.calcHist([img_resized], [2], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_resized], [0], None, [256], [0, 256]).flatten()
    
    return glcm_features, rgb_stats, hist_r, hist_g, hist_b, gray

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


# ================== DASHBOARD ==================
st.header("🔬 Webiste Jiabao Clinic")
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

    st.markdown("---")
    st.subheader("📊 Analisis Detail Gambar")
    
    # Get detailed analysis
    glcm_features, rgb_stats, hist_r, hist_g, hist_b, gray_img = analisis_detail_gambar(img)
    
    # Display analysis in tabs
    tab1, tab2, tab3 = st.tabs(["🔍 GLCM Analysis", "🎨 RGB Analysis", "🤖 Prediksi"])
    
    with tab1:
        st.subheader("Gray-Level Co-occurrence Matrix (GLCM) Features")
        
        # GLCM features table
        glcm_df = pd.DataFrame([
            {"Property": "Contrast", "0°": f"{glcm_features['contrast_0']:.4f}", 
             "45°": f"{glcm_features['contrast_45']:.4f}", "90°": f"{glcm_features['contrast_90']:.4f}", 
             "135°": f"{glcm_features['contrast_135']:.4f}"},
            {"Property": "Dissimilarity", "0°": f"{glcm_features['dissimilarity_0']:.4f}", 
             "45°": f"{glcm_features['dissimilarity_45']:.4f}", "90°": f"{glcm_features['dissimilarity_90']:.4f}", 
             "135°": f"{glcm_features['dissimilarity_135']:.4f}"},
            {"Property": "Homogeneity", "0°": f"{glcm_features['homogeneity_0']:.4f}", 
             "45°": f"{glcm_features['homogeneity_45']:.4f}", "90°": f"{glcm_features['homogeneity_90']:.4f}", 
             "135°": f"{glcm_features['homogeneity_135']:.4f}"},
            {"Property": "Energy", "0°": f"{glcm_features['energy_0']:.4f}", 
             "45°": f"{glcm_features['energy_45']:.4f}", "90°": f"{glcm_features['energy_90']:.4f}", 
             "135°": f"{glcm_features['energy_135']:.4f}"},
            {"Property": "Correlation", "0°": f"{glcm_features['correlation_0']:.4f}", 
             "45°": f"{glcm_features['correlation_45']:.4f}", "90°": f"{glcm_features['correlation_90']:.4f}", 
             "135°": f"{glcm_features['correlation_135']:.4f}"}
        ])
        
        st.dataframe(glcm_df, use_container_width=True)
        
        # GLCM visualization
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_img, caption="Grayscale Image", use_container_width=True, clamp=True)
        
        with col2:
            # GLCM properties radar chart
            angles = ['0°', '45°', '90°', '135°']
            contrast_values = [glcm_features[f'contrast_{angle}'] for angle in [0, 45, 90, 135]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=contrast_values,
                theta=angles,
                fill='toself',
                name='Contrast'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True)
                ),
                title="GLCM Contrast by Angle"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        st.subheader("RGB Color Analysis")
        
        # RGB statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 Red Channel", f"{rgb_stats['red_mean']:.1f}", f"±{rgb_stats['red_std']:.1f}")
            st.metric("Red Range", f"{rgb_stats['red_min']:.0f} - {rgb_stats['red_max']:.0f}")
        
        with col2:
            st.metric("🟢 Green Channel", f"{rgb_stats['green_mean']:.1f}", f"±{rgb_stats['green_std']:.1f}")
            st.metric("Green Range", f"{rgb_stats['green_min']:.0f} - {rgb_stats['green_max']:.0f}")
        
        with col3:
            st.metric("🔵 Blue Channel", f"{rgb_stats['blue_mean']:.1f}", f"±{rgb_stats['blue_std']:.1f}")
            st.metric("Blue Range", f"{rgb_stats['blue_min']:.0f} - {rgb_stats['blue_max']:.0f}")
        
        # RGB Histogram
        x_values = list(range(256))
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=x_values, y=hist_r, mode='lines', name='Red', line=dict(color='red')))
        fig_hist.add_trace(go.Scatter(x=x_values, y=hist_g, mode='lines', name='Green', line=dict(color='green')))
        fig_hist.add_trace(go.Scatter(x=x_values, y=hist_b, mode='lines', name='Blue', line=dict(color='blue')))
        
        fig_hist.update_layout(
            title="RGB Histogram Distribution",
            xaxis_title="Pixel Intensity",
            yaxis_title="Frequency",
            hovermode='x unified'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Color dominance pie chart
        total_intensity = rgb_stats['red_mean'] + rgb_stats['green_mean'] + rgb_stats['blue_mean']
        red_pct = (rgb_stats['red_mean'] / total_intensity) * 100
        green_pct = (rgb_stats['green_mean'] / total_intensity) * 100
        blue_pct = (rgb_stats['blue_mean'] / total_intensity) * 100
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Red', 'Green', 'Blue'],
            values=[red_pct, green_pct, blue_pct],
            marker_colors=['red', 'green', 'blue']
        )])
        fig_pie.update_layout(title="Color Channel Dominance")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab3:
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

