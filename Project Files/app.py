import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model
# Make sure 'floods.save' is in the same folder as this script
try:
    model = joblib.load('floods.save')
except:
    st.error("Model file 'floods.save' not found! Please check your folder.")

# Page Config
st.set_page_config(page_title="Flood Prediction AI", layout="centered")

st.title("🌊 Flood Risk Assessment Tool")
st.markdown("---")

# 2. Input Section (10 Features)
st.subheader("Environmental Parameters (Scale 0-10)")
st.write("Adjust the values based on current conditions:")

# Layout inputs in two columns
col1, col2 = st.columns(2)

with col1:
    v1 = st.slider("Monsoon Intensity", 0.0, 10.0, 5.0)
    v2 = st.slider("Topography Drainage", 0.0, 10.0, 5.0)
    v3 = st.slider("River Management", 0.0, 10.0, 5.0)
    v4 = st.slider("Deforestation", 0.0, 10.0, 5.0)
    v5 = st.slider("Urbanization", 0.0, 10.0, 5.0)

with col2:
    v6 = st.slider("Climate Change", 0.0, 10.0, 5.0)
    v7 = st.slider("Dams Quality", 0.0, 10.0, 5.0)
    v8 = st.slider("Siltation", 0.0, 10.0, 5.0)
    v9 = st.slider("Agricultural Practices", 0.0, 10.0, 5.0)
    v10 = st.slider("Encroachments", 0.0, 10.0, 5.0)

st.markdown("---")

# 3. Prediction Logic
if st.button("Analyze Flood Risk", use_container_width=True):
    # Prepare features for the model (Shape must be 1x10)
    features = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]])
    
    # Get Probability (This tells us WHY it might be showing 0)
    # [prob_of_0, prob_of_1]
    prob = model.predict_proba(features)[0]
    flood_chance = prob[1] * 100  # Convert to percentage
    
    # Get Hard Prediction (0 or 1)
    prediction = model.predict(features)

    # 4. Display Results
    st.subheader("Analysis Results:")
    
    # Show the percentage chance
    st.metric(label="Flood Probability", value=f"{flood_chance:.2f}%")
    
    if flood_chance > 50:
        st.error(f"🚨 ALERT: High Risk! (Confidence: {flood_chance:.1f}%)")
        st.write("The model predicts a high likelihood of flood events. Immediate precautions recommended.")
    else:
        st.success(f"✅ Safe: Low Risk (Confidence: {100 - flood_chance:.1f}%)")
        st.write("Environmental factors suggest stable conditions.")

    # Debugging Info (Hidden by default, click to expand)
    with st.expander("See Raw Model Output"):
        st.write(f"Raw Classes: {model.classes_}")
        st.write(f"Raw Probabilities: {prob}")
        st.write("Feature Vector:", features)