import pandas as pd

# Load original dataset
df = pd.read_csv("data/plant_health_data.csv")

# Select important columns
important_columns = [
    'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal', 'Plant_Health_Status'
]

# Filter dataset
df = df[important_columns]

# Save cleaned dataset
df.to_csv("data/cleaned_plant_data.csv", index=False)
print("✅ Cleaned dataset saved: data/cleaned_plant_data.csv")
print("\nFirst 5 rows:")
print(df.head())
