import pandas as pd
import joblib

# Load trained model, scaler, and label encoder
model = joblib.load("D:/Plant_Health_Classification/plant_ai_model.pkl")
scaler = joblib.load("D:/Plant_Health_Classification/scaler.pkl")
le = joblib.load("D:/Plant_Health_Classification/label_encoder.pkl")

# Example new data (simulate sensor readings)
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

# Scale features
df_new_scaled = scaler.transform(df_new)

# Predict plant health
y_pred_encoded = model.predict(df_new_scaled)
y_pred_label = le.inverse_transform(y_pred_encoded)

print("🔹 New plant data:")
print(df_new)
print("\n🔹 Predicted Plant Health Status:")
print(y_pred_label)
