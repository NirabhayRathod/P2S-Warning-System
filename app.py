# streamlit_app.py
import streamlit as st
import pandas as pd
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData

# --- Page config ---
st.set_page_config(page_title="P2S Warning System", page_icon="🌋", layout="centered")

# --- Title & description ---
st.title("🌋 P2S Warning System")
st.markdown("""
This system detects **P-wave signals** from seismic readings and predicts  
the **Time-to-Failure (TTF)** before the destructive S-wave arrives.

- **P-wave Detection** → Quick classification to detect earthquake presence.
- **TTF Prediction** → Regression model to estimate time (in seconds) until S-wave.
""")

# --- Input Form ---
st.subheader("📥 Enter Seismic Sensor Readings")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sensor_reading = st.number_input("Sensor Reading", value=2.83, format="%.2f")
        noise_level = st.number_input("Noise Level", value=0.41, format="%.2f")
        rolling_avg = st.number_input("Rolling Avg", value=2.86, format="%.2f")
    
    with col2:
        reading_diff = st.number_input("Reading Diff", value=0.05, format="%.2f")
        pga = st.number_input("PGA", value=0.51, format="%.2f")
        snr = st.number_input("SNR", value=16.86, format="%.2f")
    
    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction Logic ---
if submitted:
    try:
        obj = CustomData(sensor_reading, noise_level, rolling_avg, reading_diff, pga, snr)
        frame = obj.get_data_as_dataframe()
        
        pred_obj = PredictPipeline()
        p_wave_result = pred_obj.predict_cl(frame)
        
        if p_wave_result[0] == 1:
            st.success("✅ **P-wave detected!** ")
            ttf_result = pred_obj.predict_rg(frame)
            st.metric("⏳ Estimated Time-to-Failure (seconds)", f"{ttf_result[0]:.2f}")
        else:
            st.error("❌ No P-wave detected. No TTF prediction needed.")

    except Exception as e:
        st.error(f"Error: {e}")
