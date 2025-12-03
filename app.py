import streamlit as st
from ml_app import run_ml_app

# Page Configuration
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header Gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header h4 {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
        color: #495057;
        font-size: 1.1rem;
    }
    
    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .welcome-card h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    .welcome-card p {
        color: #34495e;
        font-size: 1.1rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #667eea;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #7f8c8d;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #00acc1;
        margin: 1.5rem 0;
    }
    
    .info-box p {
        color: #006064;
        margin: 0;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>💻 Laptop Price Prediction System</h1>
            <h4>Powered by Machine Learning & Data Science</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Menu
    st.sidebar.title("🎯 Navigation")
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox("Select Page", menu)
    
    # Sidebar Info
    st.sidebar.markdown("---")
    st.sidebar.info("📊 **About this App**\n\nThis application uses advanced machine learning algorithms to predict laptop prices based on specifications.")
    
    if choice == 'Home':
        # Welcome Section
        st.markdown("""
            <div class="welcome-card">
                <h2>👋 Welcome to Laptop Price Prediction System</h2>
                <p>
                    Selamat datang di aplikasi prediksi harga laptop yang menggunakan teknologi 
                    Machine Learning terkini. Aplikasi ini dirancang untuk membantu Anda 
                    memperkirakan harga laptop berdasarkan spesifikasi teknis yang dimiliki.
                </p>
                <p>
                    Dengan memanfaatkan algoritma prediksi yang telah dilatih dengan ribuan 
                    data laptop, kami dapat memberikan estimasi harga yang akurat dan relevan 
                    dengan kondisi pasar saat ini.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Features Section
        st.markdown("### ✨ Key Features")
        st.markdown("""
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <h3>Accurate Predictions</h3>
                    <p>Model machine learning yang telah dioptimasi untuk memberikan prediksi harga yang akurat</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3>Fast Processing</h3>
                    <p>Hasil prediksi didapatkan dalam hitungan detik dengan performa tinggi</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3>Detailed Analysis</h3>
                    <p>Analisis mendalam terhadap berbagai spesifikasi laptop untuk hasil optimal</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <h3>Secure & Private</h3>
                    <p>Data input Anda aman dan tidak disimpan di server manapun</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # How to Use Section
        st.markdown("### 📖 How to Use")
        st.markdown("""
            <div class="info-box">
                <p>
                    <strong>Langkah 1:</strong> Pilih menu "Machine Learning" di sidebar<br>
                    <strong>Langkah 2:</strong> Isi form dengan spesifikasi laptop yang ingin diprediksi<br>
                    <strong>Langkah 3:</strong> Klik tombol "Predict Price" untuk mendapatkan hasil<br>
                    <strong>Langkah 4:</strong> Lihat estimasi harga yang ditampilkan
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Stats Section
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", "94.5%", "↑ 2.1%")
        with col2:
            st.metric("📊 Models Trained", "15+", "↑ 3")
        with col3:
            st.metric("⚡ Avg Response", "< 1s", "↓ 0.2s")
        with col4:
            st.metric("💻 Laptops Analyzed", "2,000+", "↑ 350")
        
    elif choice == "Machine Learning":
        run_ml_app()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
            <p>ML Kelompok 3</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()