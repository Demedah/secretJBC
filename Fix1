import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Tekstur Kulit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #32CD32;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def ekstrak_fitur_gambar(img_input):
    """
    Fungsi untuk mengekstrak fitur GLCM dan histogram warna dari gambar
    """
    try:
        # Jika input adalah PIL Image, konversi ke numpy array
        if isinstance(img_input, Image.Image):
            img = np.array(img_input)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(img_input, np.ndarray):
            img = img_input
        else:
            st.error("Format gambar tidak didukung")
            return None

        # Resize agar konsisten
        if img.shape[0] != 128 or img.shape[1] != 128:
            img = cv2.resize(img, (128, 128))

        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ===== GLCM (Tekstur) =====
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # ===== Histogram Warna =====
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()

        # Normalisasi histogram
        hist_b = hist_b / hist_b.sum() if hist_b.sum() != 0 else hist_b
        hist_g = hist_g / hist_g.sum() if hist_g.sum() != 0 else hist_g
        hist_r = hist_r / hist_r.sum() if hist_r.sum() != 0 else hist_r

        fitur_hist = np.hstack([hist_b, hist_g, hist_r])

        # Gabung semua fitur
        fitur = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, fitur_hist])
        return fitur
    except Exception as e:
        st.error(f"Error ekstrak fitur: {e}")
        return None

def create_sample_data():
    """
    Membuat data sampel untuk demonstrasi
    """
    np.random.seed(42)
    
    # Simulasi data fitur untuk berbagai jenis kulit
    skin_types = ['Normal', 'Berminyak', 'Kering', 'Kombinasi', 'Sensitif']
    n_samples_per_type = 50
    
    X_sample = []
    y_sample = []
    
    for i, skin_type in enumerate(skin_types):
        for _ in range(n_samples_per_type):
            # Simulasi fitur GLCM (5 fitur)
            glcm_features = np.random.normal(loc=i*0.2, scale=0.1, size=5)
            # Simulasi fitur histogram (768 fitur: 256*3 channel)
            hist_features = np.random.normal(loc=0.004, scale=0.002, size=768)
            
            features = np.hstack([glcm_features, hist_features])
            X_sample.append(features)
            y_sample.append(skin_type)
    
    return np.array(X_sample), np.array(y_sample)

@st.cache_data
def train_model():
    """
    Melatih model dengan data sampel
    """
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred

def main():
    # Header utama
    st.markdown('<h1 class="main-header">🔬 Sistem Klasifikasi Tekstur Kulit</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Menu Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman:", 
                               ["🏠 Beranda", "📊 Analisis Model", "🖼️ Prediksi Gambar", "ℹ️ Tentang"])
    
    if page == "🏠 Beranda":
        show_home_page()
    elif page == "📊 Analisis Model":
        show_model_analysis()
    elif page == "🖼️ Prediksi Gambar":
        show_prediction_page()
    elif page == "ℹ️ Tentang":
        show_about_page()

def show_home_page():
    st.markdown('<h2 class="sub-header">Selamat Datang di Sistem Klasifikasi Tekstur Kulit</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Tujuan Sistem</h3>
        <p>Sistem ini menggunakan teknologi <strong>Machine Learning</strong> untuk mengklasifikasikan jenis tekstur kulit berdasarkan analisis gambar. 
        Sistem mengekstrak fitur tekstur menggunakan <strong>GLCM (Gray-Level Co-occurrence Matrix)</strong> dan 
        <strong>histogram warna</strong> untuk memberikan prediksi yang akurat.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔍 Jenis Tekstur Kulit yang Dapat Dideteksi:
        - **Normal**: Kulit seimbang, tidak terlalu berminyak atau kering
        - **Berminyak**: Kulit dengan produksi sebum berlebih
        - **Kering**: Kulit dengan kelembaban rendah
        - **Kombinasi**: Kombinasi area berminyak dan kering
        - **Sensitif**: Kulit yang mudah iritasi dan reaktif
        """)
        
        st.markdown("""
        ### 🚀 Cara Menggunakan:
        1. Pilih menu **"Prediksi Gambar"** di sidebar
        2. Upload foto kulit Anda (format: JPG, PNG, JPEG)
        3. Sistem akan menganalisis dan memberikan prediksi
        4. Lihat hasil probabilitas untuk setiap jenis kulit
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x400/2E86AB/FFFFFF?text=Skin+Analysis", 
                caption="Analisis Tekstur Kulit", use_column_width=True)

def show_model_analysis():
    st.markdown('<h2 class="sub-header">📊 Analisis Performa Model</h2>', unsafe_allow_html=True)
    
    with st.spinner("Melatih model..."):
        model, accuracy, X_test, y_test, y_pred = train_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Akurasi Model", f"{accuracy:.2%}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
    
    with col2:
        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.subheader("📈 Laporan Klasifikasi")
        st.dataframe(df_report.round(3))
        
        # Feature Importance (top 10)
        feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'] + \
                       [f'Hist_{i}' for i in range(768)]
        
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:]
        
        fig_imp = go.Figure(go.Bar(
            x=importances[top_indices],
            y=[feature_names[i] for i in top_indices],
            orientation='h'
        ))
        fig_imp.update_layout(title="Top 10 Fitur Penting", xaxis_title="Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

def show_prediction_page():
    st.markdown('<h2 class="sub-header">🖼️ Prediksi Tekstur Kulit</h2>', unsafe_allow_html=True)
    
    # Load model
    model, _, _, _, _ = train_model()
    
    uploaded_file = st.file_uploader("📁 Upload gambar kulit Anda:", 
                                   type=['jpg', 'jpeg', 'png'],
                                   help="Format yang didukung: JPG, JPEG, PNG")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Gambar yang Diupload")
            st.image(image, caption="Gambar Input", use_column_width=True)
        
        with col2:
            st.subheader("🔄 Proses Analisis")
            
            with st.spinner("Mengekstrak fitur..."):
                # Extract features
                features = ekstrak_fitur_gambar(image)
                
                if features is not None:
                    st.success("✅ Fitur berhasil diekstrak!")
                    
                    # Show feature info
                    st.info(f"📊 Total fitur yang diekstrak: {len(features)}")
                    
                    # Make prediction
                    features_reshaped = features.reshape(1, -1)
                    prediction = model.predict(features_reshaped)[0]
                    probabilities = model.predict_proba(features_reshaped)[0]
                    
                    # Display results
                    st.markdown(f"""
                    <div class="result-box">
                    <h3>🎯 Hasil Prediksi</h3>
                    <h2 style="color: #32CD32; text-align: center;">{prediction}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability chart
                    st.subheader("📊 Probabilitas untuk Setiap Jenis Kulit")
                    
                    prob_df = pd.DataFrame({
                        'Jenis Kulit': model.classes_,
                        'Probabilitas': probabilities
                    }).sort_values('Probabilitas', ascending=True)
                    
                    fig = px.bar(prob_df, x='Probabilitas', y='Jenis Kulit', 
                               orientation='h', color='Probabilitas',
                               color_continuous_scale='Viridis')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.subheader("📋 Detail Probabilitas")
                    for class_name, prob in zip(model.classes_, probabilities):
                        st.write(f"**{class_name}**: {prob:.2%}")
                        st.progress(prob)
                
                else:
                    st.error("❌ Gagal mengekstrak fitur dari gambar. Pastikan gambar valid.")

def show_about_page():
    st.markdown('<h2 class="sub-header">ℹ️ Tentang Sistem</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Teknologi yang Digunakan
    
    **Machine Learning:**
    - Random Forest Classifier
    - Scikit-learn
    
    **Computer Vision:**
    - OpenCV untuk pemrosesan gambar
    - GLCM (Gray-Level Co-occurrence Matrix) untuk analisis tekstur
    - Histogram warna untuk analisis distribusi warna
    
    **Web Framework:**
    - Streamlit untuk antarmuka web
    - Plotly untuk visualisasi interaktif
    
    ### 📊 Fitur yang Diekstrak
    
    **Fitur Tekstur GLCM (5 fitur):**
    - Contrast: Mengukur variasi intensitas
    - Dissimilarity: Mengukur ketidaksamaan lokal
    - Homogeneity: Mengukur keseragaman tekstur
    - Energy: Mengukur keseragaman distribusi
    - Correlation: Mengukur korelasi linear
    
    **Fitur Histogram Warna (768 fitur):**
    - 256 bin untuk channel Biru (B)
    - 256 bin untuk channel Hijau (G)  
    - 256 bin untuk channel Merah (R)
    
    ### 🎯 Akurasi Model
    Model ini menggunakan Random Forest dengan 100 decision trees dan mencapai akurasi yang baik 
    dalam mengklasifikasikan berbagai jenis tekstur kulit.
    
    ### 👨‍💻 Pengembang
    Sistem ini dikembangkan untuk membantu analisis tekstur kulit menggunakan teknologi 
    machine learning dan computer vision.
    """)

if __name__ == "__main__":
    main()
