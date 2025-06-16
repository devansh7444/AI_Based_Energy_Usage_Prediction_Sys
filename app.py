import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
import os

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="Energy Usage Optimizer", layout="wide")
st.title("🔋 AI-Powered Energy Usage Optimization App")

st.markdown("""
This app analyzes your appliance-wise energy consumption and provides:
- 📈 Forecasted usage trends
- 🚨 Alerts for abnormal energy patterns
- 🌱 Tips to reduce energy waste

**Model trained on UCI Power Consumption Dataset**
""")

# Upload CSV file
data_file = st.file_uploader("📂 Upload your energy consumption data (CSV, last 60 days)", type=["csv"])

# Appliance Mapping and Model Loader
appliance_models = {
    "Kitchen (Sub_metering_1)": "model_kitchen.h5",
    "Laundry (Sub_metering_2)": "model_laundry.h5",
    "AC/Heater (Sub_metering_3)": "model_ac_heater.h5"
}

if data_file:
    df = pd.read_csv(data_file, parse_dates=True, index_col=0)
    df = df.dropna()

    st.subheader("Raw Data Preview")
    st.dataframe(df.tail())

    # Select appliance
    selected_appliance = st.selectbox("Select Appliance:", list(appliance_models.keys()))
    col = selected_appliance.split("(")[1].replace(")", "")

    st.line_chart(df[col])

    # Preprocessing
    values = df[[col]].values[-30:]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X_input = np.reshape(scaled, (1, 30, 1))

    # Load model and predict
    model_path = appliance_models[selected_appliance]
    if os.path.exists(model_path):
        model = load_model(
            model_path,
            custom_objects={
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError()
            }
        )
        pred_scaled = model.predict(X_input)
        pred_actual = scaler.inverse_transform(pred_scaled)
        st.metric("⚡ Predicted Next-Day Usage", f"{pred_actual[0][0]:.2f} kWh")
    else:
        st.warning(f"Model file '{model_path}' not found. Please upload or train the model.")

    # Basic abnormal usage alert
    st.subheader("🚨 Usage Pattern Check")
    threshold = st.slider("Set average threshold (kWh)", 1.0, 50.0, 10.0)
    avg_val = df[col].tail(7).mean()
    if avg_val > threshold:
        st.error(f"High recent usage: {avg_val:.2f} kWh")
    else:
        st.success(f"Usage in control: {avg_val:.2f} kWh")

    # Tips
    st.subheader("🌿 Green Energy Tips")
    tips = {
        "Sub_metering_1": "Run kitchen appliances in off-peak hours.",
        "Sub_metering_2": "Use cold water wash and energy saver modes.",
        "Sub_metering_3": "Use thermostat timers and insulation to reduce heating/cooling demand."
    }
    st.info(tips.get(col, "Use energy-efficient appliances and switch off unused devices."))

else:
    st.info("Please upload energy data (preferably 30+ recent days).")
