import streamlit as st
import pandas as pd
import joblib
import re

def run_ml_app():

    def FE_manual(df):
        df = df.copy()

        # =====================================================
        # 1. DROP PRODUCT
        # =====================================================
        if "Product" in df.columns:
            df.drop("Product", axis=1, inplace=True)

        # =====================================================
        # 2. CPU FEATURE ENGINEERING
        # =====================================================
        df["cpu_series"] = df["CPU_Type"].str.extract(
            r'(Core i3|Core i5|Core i7|Core i9|Core M|Pentium|Celeron|Atom|Ryzen \d|Ryzen|Xeon|A\d-Series|FX)',
            expand=False
        ).fillna("Other")

        def classify_cpu(cpu):
            cpu = str(cpu).upper()

            if "XEON" in cpu:
                return "Workstation"

            if ("I7" in cpu or "I9" in cpu) and any(x in cpu for x in ["HQ","HK","H"]):
                return "High"
            if "RYZEN 7" in cpu or "RYZEN 9" in cpu:
                return "High"
            if "FX 8" in cpu or "FX 9" in cpu:
                return "High"

            if "RYZEN 5" in cpu:
                return "Mid"
            if any(x in cpu for x in ["CORE M", "M3", "M5", "M7"]):
                return "Mid"
            if "I5" in cpu:
                return "Mid"
            if "I7" in cpu and any(x in cpu for x in ["U","Y"]):
                return "Mid"

            if "I3" in cpu:
                return "Entry"
            if any(x in cpu for x in ["PENTIUM", "CELERON"]):
                return "Entry"
            if any(x in cpu for x in ["A4","A6","A8","A9","A10","A12"]):
                return "Entry"
            if "E-SERIES" in cpu or "E2" in cpu:
                return "Entry"

            if "ATOM" in cpu:
                return "LowPower"
            if "Y" in cpu and not ("I5" in cpu or "I7" in cpu):
                return "LowPower"

            if "CORTEX" in cpu:
                return "ARM"

            return "Entry"

        df["cpu_class"] = df["CPU_Type"].apply(classify_cpu)
        df.drop("CPU_Type", axis=1, inplace=True)

        # =====================================================
        # 3. GPU FEATURE ENGINEERING
        # =====================================================
        def classify_gpu(gpu):
            gpu = str(gpu).upper()

            if any(x in gpu for x in ["MALI", "ADRENO", "POWERVR"]):
                return "ARM_GPU"

            if "QUADRO" in gpu:
                return "NVIDIA_Workstation"
            if "FIREPRO" in gpu:
                return "AMD_Workstation"

            if "GTX" in gpu:
                return "NVIDIA_Gaming"

            if "MX" in gpu or any(x in gpu for x in ["920M", "930M", "940M"]):
                return "NVIDIA_Entry"

            if "RX" in gpu:
                return "AMD_RX"

            if any(x in gpu for x in ["R5", "R7", "R9"]):
                return "AMD_Dedicated"

            if "IRIS" in gpu:
                return "Integrated_Premium"

            if "HD GRAPHICS" in gpu or "UHD" in gpu:
                return "Integrated_Basic"
            if any(x in gpu for x in ["R2","R3","R4","R5"]):
                return "Integrated_Basic"

            return "Other"

        df["gpu_class"] = df["GPU_Type"].apply(classify_gpu)
        df.drop("GPU_Type", axis=1, inplace=True)

        # =====================================================
        # 4. SCREEN FEATURE ENGINEERING
        # =====================================================
        df["is_touchscreen"] = df["ScreenResolution"].str.contains("touch", case=False).astype(int)

        df["res"] = df["ScreenResolution"].str.extract(r'(\d{3,4}x\d{3,4})')
        df["res_x"] = df["res"].str.split("x").str[0].astype(float)
        df["res_y"] = df["res"].str.split("x").str[1].astype(float)

        df["panel_type"] = (
            df["ScreenResolution"]
            .str.extract(r'(IPS Panel|Retina Display|Retina|TN|OLED|AMOLED)', expand=False)
            .fillna("Other")
        )

        def res_category(x):
            if pd.isna(x): return "Other"
            if x >= 3840: return "4K"
            if x >= 2560: return "QHD"
            if x >= 1920: return "FullHD"
            if x >= 1366: return "HD"
            return "Other"

        df["res_category"] = df["res_x"].apply(res_category)
        df["megapixel"] = (df["res_x"] * df["res_y"]) / 1e6

        df.drop(["res", "ScreenResolution"], axis=1, inplace=True)

        # =====================================================
        # 5. MEMORY FEATURE ENGINEERING
        # =====================================================
        def parse_memory(mem):
            mem_raw = mem.lower()
            mem_clean = mem.replace(" ", "").upper()

            ssd = hdd = hybrid = flash = 0
            parts = re.split(r'\+', mem_clean)

            for idx, part in enumerate(parts):
                tb = re.search(r'(\d+\.?\d*)TB', part)
                gb = re.search(r'(\d+\.?\d*)GB', part)

                if tb:
                    size_gb = float(tb.group(1)) * 1024
                elif gb:
                    size_gb = float(gb.group(1))
                else:
                    size_gb = 0

                raw = mem_raw.split('+')[idx]

                if "ssd" in raw:
                    ssd += size_gb
                elif "hdd" in raw:
                    hdd += size_gb
                elif "hybrid" in raw:
                    hybrid += size_gb
                elif "flash" in raw:
                    flash += size_gb

            return pd.Series([ssd, hdd, hybrid, flash])

        df[["total_ssd_gb", "total_hdd_gb", "total_hybrid_gb", "total_flash_gb"]] = \
            df["Memory"].apply(parse_memory)

        def main_storage_type(row):
            if row.total_ssd_gb > 0: return "SSD"
            if row.total_hdd_gb > 0: return "HDD"
            if row.total_hybrid_gb > 0: return "Hybrid"
            if row.total_flash_gb > 0: return "Flash"
            return "Unknown"

        df["main_storage"] = df.apply(main_storage_type, axis=1)

        df.drop("Memory", axis=1, inplace=True)

        return df

    # Custom CSS untuk form styling
    st.markdown("""
        <style>
        /* Form Container */
        .stForm {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        /* Section Headers */
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0 1rem 0;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        }
        
        /* Input Fields */
        .stTextInput input, .stNumberInput input {
            border-radius: 8px;
            border: 2px solid #e9ecef;
            padding: 0.75rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        /* Labels */
        .stTextInput label, .stNumberInput label {
            font-weight: 500;
            color: #495057;
            margin-bottom: 0.5rem;
        }
        
        /* Submit Button */
        .stFormSubmitButton button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 1rem;
            border-radius: 10px;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(102,126,234,0.3);
        }
        
        .stFormSubmitButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(102,126,234,0.4);
        }
        
        /* Success Message */
        .success-box {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 5px solid #28a745;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            animation: slideIn 0.5s ease;
        }
        
        .success-box h3 {
            color: #155724;
            margin: 0;
            font-size: 1.5rem;
        }
        
        .price-amount {
            font-size: 2.5rem;
            font-weight: 700;
            color: #28a745;
            margin-top: 0.5rem;
        }
        
        @keyframes slideIn {
            from { 
                opacity: 0; 
                transform: translateX(-20px); 
            }
            to { 
                opacity: 1; 
                transform: translateX(0); 
            }
        }
        
        /* Info Card */
        .info-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #2196f3;
            margin: 1.5rem 0;
        }
        
        .info-card p {
            color: #0d47a1;
            margin: 0;
            font-weight: 500;
        }
        
        /* Example Hint */
        .example-hint {
            font-size: 0.85rem;
            color: #6c757d;
            font-style: italic;
            margin-top: 0.25rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title with icon
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #667eea; font-weight: 700;">💻 Laptop Price Prediction</h1>
            <p style="color: #6c757d; font-size: 1.1rem;">Enter laptop specifications to get accurate price estimation</p>
        </div>
    """, unsafe_allow_html=True)

    # Info Card
    st.markdown("""
        <div class="info-card">
            <p>📝 Fill in all the specifications below. You can use examples provided for reference.</p>
        </div>
    """, unsafe_allow_html=True)

    # ============================
    # 1. INPUT FORM
    # ============================

    with st.form("input_form"):

        # Basic Information Section
        st.markdown('<div class="section-header">🏢 Basic Information</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            Company = st.text_input("Company", placeholder="e.g., Dell, HP, Lenovo")
        with col2:
            Product = st.text_input("Product", placeholder="e.g., Inspiron 15, ThinkPad")
        with col3:
            TypeName = st.text_input("Type Name", placeholder="e.g., Ultrabook, Gaming")

        # Display Specifications Section
        st.markdown('<div class="section-header">🖥️ Display Specifications</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            Inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, step=0.1, value=15.6)
        with col2:
            ScreenResolution = st.text_input(
                "Screen Resolution",
                placeholder="IPS Panel Retina Display 2560x1600"
            )

        # Processor Section
        st.markdown('<div class="section-header">⚙️ Processor Specifications</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            CPU_Company = st.text_input("CPU Company", placeholder="Intel, AMD")
        with col2:
            CPU_Type = st.text_input("CPU Type", placeholder="Core i5 7200U")
        with col3:
            CPU_Frequency = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=5.0, step=0.1, value=2.5)

        # Memory & Storage Section
        st.markdown('<div class="section-header">💾 Memory & Storage</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            RAM = st.number_input("RAM (GB)", min_value=2, max_value=128, step=2, value=8)
        with col2:
            Memory = st.text_input(
                "Storage Configuration",
                placeholder="256GB SSD + 1TB HDD"
            )

        # Graphics & System Section
        st.markdown('<div class="section-header">🎮 Graphics & System</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            GPU_Company = st.text_input("GPU Company", placeholder="Intel, Nvidia, AMD")
            GPU_Type = st.text_input("GPU Type", placeholder="GTX 1050, Iris Plus")
        with col2:
            OpSys = st.text_input("Operating System", placeholder="Windows 10, macOS")
            Weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.01, value=2.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Predict Price")


    # ============================
    # 2. PREDICTION PIPELINE
    # ============================

    if submitted:
        
        # Validation
        if not all([Company, TypeName, ScreenResolution, CPU_Company, CPU_Type, Memory, GPU_Company, GPU_Type, OpSys]):
            st.error("⚠️ Please fill in all required fields!")
            return

        with st.spinner("🔄 Processing your request..."):
            
            input_data = pd.DataFrame([{
                "Company": Company,
                "Product": Product,
                "TypeName": TypeName,
                "Inches": Inches,
                "ScreenResolution": ScreenResolution,
                "CPU_Company": CPU_Company,
                "CPU_Type": CPU_Type,
                "CPU_Frequency (GHz)": CPU_Frequency,
                "RAM (GB)": RAM,
                "Memory": Memory,
                "GPU_Company": GPU_Company,
                "GPU_Type": GPU_Type,
                "OpSys": OpSys,
                "Weight (kg)": Weight
            }])

            # Show input data in expandable section
            with st.expander("📋 View Input Data", expanded=False):
                st.dataframe(input_data, use_container_width=True)

            try:
                # ========== Load Final Pipeline ==========
                input_data_fe = FE_manual(input_data)
                preprocessor = joblib.load("preprocessor.pkl")
                input_data_processed = preprocessor.transform(input_data_fe)
                model = joblib.load("model.pkl")
                prediction = model.predict(input_data_processed)[0]

                # Success Message with styled result
                st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Prediction Successful!</h3>
                        <div style="margin-top: 1rem;">
                            <p style="color: #155724; font-size: 1.1rem; margin-bottom: 0.5rem;">
                                Estimated Price:
                            </p>
                            <div class="price-amount">€{prediction:,.2f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"💰 Price Range: €{prediction*0.9:,.2f} - €{prediction*1.1:,.2f}")
                with col2:
                    st.info(f"📊 Confidence: High")
                with col3:
                    st.info(f"⏱️ Processed in < 1s")
                
            except FileNotFoundError:
                st.error("⚠️ Model files not found! Please ensure 'preprocessor.pkl' and 'model.pkl' are in the same directory.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")