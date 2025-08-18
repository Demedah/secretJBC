# uji_citra.py

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
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style untuk matplotlib
plt.style.use('default')
sns.set_palette("husl")

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
        return fitur, (contrast, dissimilarity, homogeneity, energy, correlation), (hist_b, hist_g, hist_r), gray
    except:
        return None

def visualize_glcm_features(glcm_features, title="GLCM Features"):
    """Visualisasi fitur GLCM dalam bentuk radar chart"""
    fig = go.Figure()
    
    categories = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']
    values = list(glcm_features) + [glcm_features[0]]  # Close the radar chart
    categories_closed = categories + [categories[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories_closed,
        fill='toself',
        name='GLCM Features',
        line_color='rgb(32, 201, 151)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(glcm_features) * 1.1]
            )),
        showlegend=True,
        title=title
    )
    
    return fig

def visualize_color_histogram(hist_b, hist_g, hist_r, title="Color Histogram"):
    """Visualisasi histogram warna RGB"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RGB Histogram', 'Blue Channel', 'Green Channel', 'Red Channel'),
        specs=[[{"colspan": 2}, None],
               [{}, {}]]
    )
    
    x = np.arange(256)
    
    # Combined RGB histogram
    fig.add_trace(go.Scatter(x=x, y=hist_b, name='Blue', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=hist_g, name='Green', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=hist_r, name='Red', line=dict(color='red')), row=1, col=1)
    
    # Individual channels
    fig.add_trace(go.Scatter(x=x, y=hist_b, name='Blue', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=hist_g, name='Green', line=dict(color='green'), showlegend=False), row=2, col=2)
    
    fig.update_layout(height=600, title_text=title)
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Visualisasi confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=class_names,
                    y=class_names,
                    color_continuous_scale='Blues',
                    text_auto=True)
    
    fig.update_layout(title="Confusion Matrix")
    return fig

# ============ 2) Load Dataset & Augmentasi ============
st.title("🔬 Klasifikasi Jenis Kulit Wajah - Jiabao Clinic")
st.markdown("---")

st.sidebar.title("📊 Visualization Controls")
show_augmentation = st.sidebar.checkbox("Show Data Augmentation", value=True)
show_feature_analysis = st.sidebar.checkbox("Show Feature Analysis", value=True)
show_model_performance = st.sidebar.checkbox("Show Model Performance", value=True)

# Check if dataset exists
if os.path.exists("databaseJBC.csv"):
    df = pd.read_csv("databaseJBC.csv")
    st.success(f"✅ Dataset loaded: {len(df)} samples")
    
    if show_feature_analysis:
        st.subheader("📈 Dataset Distribution")
        fig_dist = px.histogram(df, x="Tekstur Kulit", title="Distribution of Skin Types")
        st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.error("❌ Dataset 'databaseJBC.csv' not found!")
    st.stop()

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
glcm_features_list = []
sample_images = []

progress_bar = st.progress(0)
status_text = st.empty()

for idx, row in df.iterrows():
    progress_bar.progress((idx + 1) / len(df))
    status_text.text(f'Processing image {idx + 1}/{len(df)}: {row["FotoCS"]}')
    
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
        fitur_asli, glcm_vals, hist_vals, gray_img = result
        X.append(fitur_asli)
        y.append(label)
        glcm_features_list.append(glcm_vals)
        
        # Simpan beberapa sample untuk visualisasi
        if len(sample_images) < 6:
            sample_images.append({
                'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                'gray': gray_img,
                'label': label,
                'glcm': glcm_vals,
                'hist': hist_vals
            })

    # augmentasi (5 variasi per gambar)
    if show_augmentation and len(sample_images) <= 3:
        img_expanded = np.expand_dims(img, axis=0)
        aug_iter = datagen.flow(img_expanded, batch_size=1)
        aug_samples = []
        
        for i in range(5):
            aug_img = next(aug_iter)[0].astype("uint8")
            result_aug = ekstrak_fitur_gambar(aug_img)
            if result_aug is not None:
                fitur_aug, _, _, _ = result_aug
                X.append(fitur_aug)
                y.append(label)
                
                if i < 3:  # Simpan 3 sample augmentasi
                    aug_samples.append(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        
        if aug_samples and len(sample_images) <= 3:
            sample_images[-1]['augmented'] = aug_samples

progress_bar.empty()
status_text.empty()

X = np.array(X, dtype=float)
y = np.array(y)

st.success(f"✅ Data processing complete: {X.shape[0]} samples with {X.shape[1]} features")

if show_augmentation and sample_images:
    st.subheader("🖼️ Sample Images and Data Augmentation")
    
    for i, sample in enumerate(sample_images[:3]):
        st.write(f"**Sample {i+1}: {sample['label']}**")
        
        cols = st.columns([1, 1, 1, 1, 1])
        
        with cols[0]:
            st.image(sample['original'], caption="Original", use_column_width=True)
        
        with cols[1]:
            st.image(sample['gray'], caption="Grayscale", use_column_width=True)
        
        if 'augmented' in sample:
            for j, aug_img in enumerate(sample['augmented'][:3]):
                with cols[j+2]:
                    st.image(aug_img, caption=f"Aug {j+1}", use_column_width=True)

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
    
    if show_model_performance:
        st.subheader("📊 Model Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Accuracy", f"{acc:.3f}", f"{acc*100:.1f}%")
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3))
        
        with col2:
            # Confusion Matrix
            fig_cm = plot_confusion_matrix(y_test, y_pred, le.classes_)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'] + \
                       [f'Hist_B_{i}' for i in range(256)] + \
                       [f'Hist_G_{i}' for i in range(256)] + \
                       [f'Hist_R_{i}' for i in range(256)]
        
        importances = model.feature_importances_
        
        # Top 20 most important features
        top_indices = np.argsort(importances)[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        fig_importance = px.bar(
            x=top_importances, 
            y=top_features, 
            orientation='h',
            title="Top 20 Most Important Features",
            labels={'x': 'Importance', 'y': 'Features'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)

else:
    model, le = None, None
    acc = 0.0
    st.error("❌ Insufficient data for training!")

# ============ 4) Streamlit UI ============
st.markdown("---")
st.subheader("🔍 Test Your Image")

uploaded_file = st.file_uploader("Upload Foto Wajah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            use_column_width=True
        )

    result = ekstrak_fitur_gambar(img)
    if result is not None:
        fitur, glcm_vals, hist_vals, gray_img = result
        expected_len = model.n_features_in_

        if len(fitur) > expected_len:
            fitur = fitur[:expected_len]
        elif len(fitur) < expected_len:
            fitur = list(fitur) + [0.0] * (expected_len - len(fitur))

        pred_encoded = model.predict([fitur])[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        probs = model.predict_proba([fitur])[0]

        with col2:
            st.success(f"🎯 **Predicted Skin Type: {pred_label}**")
            
            # Probability distribution
            prob_df = pd.DataFrame({
                "Skin Type": le.classes_,
                "Probability": probs
            }).sort_values("Probability", ascending=False)

            fig_prob = px.bar(
                prob_df, 
                x="Probability", 
                y="Skin Type", 
                orientation='h',
                title="Prediction Confidence",
                color="Probability",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        st.subheader("🔬 Detailed Feature Analysis")
        
        tab1, tab2, tab3 = st.tabs(["GLCM Features", "Color Histogram", "Image Processing"])
        
        with tab1:
            fig_glcm = visualize_glcm_features(glcm_vals, "GLCM Features of Uploaded Image")
            st.plotly_chart(fig_glcm, use_container_width=True)
            
            # GLCM values table
            glcm_df = pd.DataFrame({
                'Feature': ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'],
                'Value': glcm_vals
            })
            st.dataframe(glcm_df, use_container_width=True)
        
        with tab2:
            hist_b, hist_g, hist_r = hist_vals
            fig_hist = visualize_color_histogram(hist_b, hist_g, hist_r, "Color Histogram of Uploaded Image")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Color statistics
            color_stats = pd.DataFrame({
                'Channel': ['Blue', 'Green', 'Red'],
                'Mean': [np.mean(hist_b), np.mean(hist_g), np.mean(hist_r)],
                'Std': [np.std(hist_b), np.std(hist_g), np.std(hist_r)],
                'Max': [np.max(hist_b), np.max(hist_g), np.max(hist_r)]
            })
            st.dataframe(color_stats, use_container_width=True)
        
        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            with col_b:
                st.image(gray_img, caption="Grayscale Image", use_column_width=True)

    else:
        st.error("⚠️ Failed to extract features from uploaded image.")

st.markdown("---")
st.markdown("""
### 📋 About This Application
- **Feature Extraction**: GLCM (Gray-Level Co-occurrence Matrix) + Color Histogram
- **Classification**: Random Forest with 100 estimators
- **Data Augmentation**: Rotation, shift, shear, zoom, and flip
- **Visualization**: Interactive charts using Plotly and Matplotlib

**GLCM Features Explained:**
- **Contrast**: Measures local variations in the image
- **Dissimilarity**: Measures how different adjacent pixels are
- **Homogeneity**: Measures closeness of distribution to diagonal
- **Energy**: Measures uniformity of texture
- **Correlation**: Measures linear dependency of gray levels
""")
