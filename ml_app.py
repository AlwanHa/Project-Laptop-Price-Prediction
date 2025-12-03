import streamlit as st
import pandas as pd
import joblib
import re

# ============================================================
# 1. FEATURE ENGINEERING MANUAL (FE_manual)
# ============================================================

def FE_manual(df):
    df = df.copy()

    # LOWERCASE kolom kategorik
    df["company"] = df["Company"].str.lower()
    df["typename"] = df["TypeName"].str.lower()
    df["cpu_company"] = df["CPU_Company"].str.lower()
    df["gpu_company"] = df["GPU_Company"].str.lower()
    df["opsys"] = df["OpSys"].str.lower()

    # ================= SCREEN RESOLUTION =================
    df["is_touchscreen"] = df["ScreenResolution"].str.contains("touch", case=False).astype(int)

    df[['screen_width', 'screen_height']] = df["ScreenResolution"].str.extract(r'(\d+)[xX](\d+)')
    df['screen_width'] = df['screen_width'].astype(int)
    df['screen_height'] = df['screen_height'].astype(int)

    # ================= MEMORY PARSER =====================
    def clean_memory_all(value):
        text = value.lower()
        num = 0.0

        # ambil angka
        m = re.search(r"(\d+)", text)
        if m:
            num = float(m.group(1))

        # jika TB → convert ke GB
        if "tb" in text:
            num *= 1024

        # storage type
        if "ssd" in text:
            stype = "ssd"
        elif "hdd" in text:
            stype = "hdd"
        elif "flash" in text:
            stype = "flash"
        elif "hybrid" in text:
            stype = "hybrid"
        else:
            stype = "others"

        return pd.Series([num, stype])

    df[['total_storage_gb', 'storage_type']] = df["Memory"].apply(clean_memory_all)

    # ================= CPU FAMILY + GEN ==================
    def extract_cpu(cpu):
        text = cpu.lower()

        # family
        if "core" in text:
            fam = "Core"
        elif "pentium dual core" in text:
            fam = "Pentium Dual Core"
        elif "pentium" in text:
            fam = "Pentium"
        elif "ryzen" in text:
            fam = "Ryzen"
        elif "fx" in text:
            fam = "FX"
        elif "cortex" in text:
            fam = "Cortex"
        else:
            fam = "Others"

        # gen
        gen = "Others"
        m = re.search(r"i[3-9]", text)
        if m:
            gen = m.group(0).upper()
        elif "ryzen" in text:
            m = re.search(r"ryzen\s*\d", text)
            gen = m.group(0).title() if m else "Ryzen"
        elif "fx" in text:
            gen = "FX"
        elif "cortex" in text:
            m = re.search(r"a\d+", text)
            gen = m.group(0).upper() if m else "Cortex"

        return fam, gen


    df[['cpu_family', 'cpu_gen_family']] = df["CPU_Type"].apply(lambda x: pd.Series(extract_cpu(x)))

    # ================= GPU FAMILY =================
    df["GPU_Family"] = df["GPU_Type"].apply(lambda x: x.split()[0])

    def extract_gpu_number(x):
        nums = re.findall(r'\d+', x)
        return int(nums[0]) if nums else None

    df["GPU_Model_Number"] = df["GPU_Type"].apply(extract_gpu_number)
    df["GPU_Model_Number"] = df["GPU_Model_Number"].fillna(0)

    df["gpu_family"] = df["GPU_Family"].str.lower()

    # drop exact same cols as training
    df = df.drop(columns=["GPU_Family", "GPU_Model_Number", "GPU_Type"])

    return df

# ============================================================
# 2. FIX FINAL COLUMNS (FE_fix)
# ============================================================

def FE_fix(df):
    df = FE_manual(df).copy()

    # rename supaya sama dengan training pipeline
    df = df.rename(columns={
        "Inches": "inches",
        "CPU_Frequency (GHz)": "cpu_frequency (ghz)",
        "RAM (GB)": "ram (gb)",
        "Weight (kg)": "weight (kg)",
    })

    final_cols = [
        "company", "typename", "inches", "cpu_company",
        "cpu_frequency (ghz)", "ram (gb)", "gpu_company", "opsys",
        "weight (kg)", "total_storage_gb", "storage_type",
        "is_touchscreen", "screen_width", "screen_height",
        "cpu_family", "cpu_gen_family", "gpu_family"
    ]

    return df[final_cols]


# ============================================================
# 3. STREAMLIT APP
# ============================================================

def run_ml_app():

    st.title("💻 Laptop Price Prediction")

    with st.form("input_form"):
        gpu_family_options = [
            'iris', 'hd', 'radeon', 'geforce', 'uhd', 'r4', 'gtx',
            'r17m-m1-70', 'quadro', 'firepro', 'graphics', 'mali'
        ]
        company_options = [
            'apple', 'hp', 'acer', 'asus', 'dell', 'lenovo', 'chuwi', 'msi',
            'microsoft', 'toshiba', 'huawei', 'xiaomi', 'vero', 'razer',
            'mediacom', 'samsung', 'google', 'fujitsu', 'lg'
        ]

        typename_options = [
            'ultrabook', 'notebook', 'netbook', 'gaming',
            '2 in 1 convertible', 'workstation'
        ]

        gpu_company_options = [
            'intel', 'amd', 'nvidia', 'arm'
        ]

        opsys_options = [
            'macos', 'no os', 'windows 10', 'mac os x', 'linux', 'android',
            'windows 10 s', 'chrome os', 'windows 7'
        ]
        # 🏢 Basic Information
        st.subheader("🏢 Basic Information")
        col1, col2, col3 = st.columns(3)

        with col1:    
            Company = st.selectbox(
                "Company",
                options=company_options,
                index=company_options.index("asus")   # default
            )

        with col2:
            Product = st.text_input("Product", "Inspiron 15")

        with col3:    
            TypeName = st.selectbox(
                "Type Name",
                options=typename_options,
                index=typename_options.index("notebook")   # default
            )


        # 🖥️ Display
        st.subheader("🖥️ Display")
        col1, col2 = st.columns(2)

        with col1:
            Inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, 15.6)

        with col2:
            ScreenResolution = st.text_input("Screen Resolution", "1920x1080")


        # ⚙ Processor
        st.subheader("⚙ Processor")
        col1, col2, col3 = st.columns(3)

        with col1:
            CPU_Company = st.text_input("CPU Company", "Intel")

        with col2:
            CPU_Type = st.text_input("CPU Type", "Core i5 7200U")

        with col3:
            CPU_Frequency = st.number_input("CPU Frequency (GHz)", 0.5, 5.0, 2.5)


        # 💾 Memory
        st.subheader("💾 Memory")
        col1, col2 = st.columns(2)

        with col1:
            RAM = st.number_input("RAM (GB)", 2, 128, 8)

        with col2:
            Memory = st.text_input("Storage Configuration", "256GB SSD")


        # 🎮 Graphics & System
        st.subheader("🎮 Graphics & System")
        col1, col2, col3 = st.columns(3)

        with col1:
            GPU_Company = st.text_input("GPU Company", "Intel")

        with col2:
            GPU_Type = st.selectbox(
                "GPU Family",
                options=gpu_family_options,
                index=gpu_family_options.index("hd") if "hd" in gpu_family_options else 0
            )

        with col3:
            OpSys = st.selectbox(
                "Operating System",
                options=opsys_options,
                index=opsys_options.index("windows 10")   # default
            )

        # Last field: Weight
        Weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0)


        submitted = st.form_submit_button("🚀 Predict Price")

    if submitted:

        raw_df = pd.DataFrame([{
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

        st.subheader("📄 Input Data Preview")
        st.dataframe(raw_df)

        try:
            processed = FE_fix(raw_df)

            model = joblib.load("model_final.pkl")
            pred = model.predict(processed)[0]

            st.success("Prediction Successful!")
            st.metric("Estimated Price (€)", f"{pred:,.2f}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
