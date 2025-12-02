import streamlit as st
import pandas as pd
import joblib

# --- Load model ---
model = joblib.load("model_final.pkl")

st.title("Laptop Price Prediction 💻")

st.markdown("Masukkan spesifikasi laptop untuk memprediksi harga.")

# --- Input form ---
with st.form("laptop_form"):
    company = st.selectbox("Company", ["Dell", "HP", "Acer", "Lenovo", "Apple"])
    typename = st.text_input("TypeName", "Notebook")
    inches = st.number_input("Screen Size (inches)", 10.0, 20.0, 15.6)
    cpu_company = st.selectbox("CPU Company", ["Intel", "AMD", "Apple"])
    cpu_frequency = st.number_input("CPU Frequency (GHz)", 1.0, 5.0, 2.5)
    ram = st.number_input("RAM (GB)", 2, 64, 8)
    gpu_company = st.selectbox("GPU Company", ["Intel", "NVIDIA", "AMD"])
    opsys = st.selectbox("Operating System", ["Windows", "MacOS", "Linux", "Others"])
    weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0)
    total_storage_gb = st.number_input("Total Storage (GB)", 128, 2000, 512)
    storage_type = st.selectbox("Storage Type", ["HDD", "SSD", "Hybrid", "Flash"])
    is_touchscreen = st.checkbox("Touchscreen")
    screen_width = st.number_input("Screen Width", 800, 3840, 1920)
    screen_height = st.number_input("Screen Height", 600, 2160, 1080)
    cpu_family = st.selectbox("CPU Family", ["Core", "Others", "Ryzen", "FX", "Cortex"])
    cpu_gen_family = st.selectbox("CPU Gen Family", ["I3","I5","I7","FX","A72","Ryzen 1","Others"])
    cpu_gen_numeric = st.number_input("CPU Gen Numeric", 1000, 10000, 7700)
    cpu_model_numeric = st.number_input("CPU Model Numeric", 1000, 10000, 7700)
    gpu_family = st.selectbox("GPU Family", ["mali", "r4", "hd", "uhd", "iris", "graphics",
                                            "radeon", "geforce", "gtx", "firepro", "quadro", "r17m-m1-70"])
    gpu_model_number = st.number_input("GPU Model Number", 100, 2000, 1050)

    submit_button = st.form_submit_button(label="Predict Price")

# --- Predict ---
if submit_button:
    input_df = pd.DataFrame([{
        "company": company,
        "typename": typename,
        "inches": inches,
        "cpu_company": cpu_company,
        "cpu_frequency (ghz)": cpu_frequency,
        "ram (gb)": ram,
        "gpu_company": gpu_company,
        "opsys": opsys,
        "weight (kg)": weight,
        "total_storage_gb": total_storage_gb,
        "storage_type": storage_type,
        "is_touchscreen": int(is_touchscreen),
        "screen_width": screen_width,
        "screen_height": screen_height,
        "cpu_family": cpu_family,
        "cpu_gen_family": cpu_gen_family,
        "cpu_gen_numeric": cpu_gen_numeric,
        "cpu_model_numeric": cpu_model_numeric,
        "gpu_family": gpu_family,
        "gpu_model_number": gpu_model_number
    }])
    
    pred_price = model.predict(input_df)[0]
    st.success(f"Predicted Laptop Price: €{pred_price:,.2f}")
