import pandas as pd
import joblib
import requests  # لو هتبعت البيانات للـ Backend API

# -----------------------------
# 1️⃣ Load the trained model, scaler, and label encoder
# -----------------------------
model = joblib.load("plant_ai_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# 2️⃣ Example new data from sensors
# Replace these values with actual sensor readings
# -----------------------------
new_data = {
    'Soil_Moisture': [30],
    'Ambient_Temperature': [25],
    'Soil_Temperature': [24],
    'Humidity': [60],
    'Light_Intensity': [500],
    'Soil_pH': [6.5],
    'Nitrogen_Level': [30],
    'Phosphorus_Level': [20],
    'Potassium_Level': [35],
    'Chlorophyll_Content': [40],
    'Electrochemical_Signal': [0.9]
}

df_new = pd.DataFrame(new_data)

# -----------------------------
# 3️⃣ Scale the data
# -----------------------------
df_new_scaled = scaler.transform(df_new)

# -----------------------------
# 4️⃣ Predict Plant Health Status
# -----------------------------
y_pred_encoded = model.predict(df_new_scaled)
y_pred_label = le.inverse_transform(y_pred_encoded)

print("🔹 Predicted Plant Health Status:")
print(list(y_pred_label))

# -----------------------------
# 5️⃣ Optional: Send result to Backend API
# -----------------------------
# Replace this URL with your actual backend endpoint
api_url = "http://localhost:5000/api/plant_status"

payload = {
    "sensor_data": new_data,
    "predicted_status": list(y_pred_label)
}

try:
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        print("✅ Data sent to backend successfully!")
    else:
        print(f"⚠ Backend returned status code {response.status_code}")
except Exception as e:
    print(f"❌ Failed to send data to backend: {e}")
