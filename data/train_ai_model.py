import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_plant_data.csv")

# Encode target column
le = LabelEncoder()
df['Plant_Health_Status_Encoded'] = le.fit_transform(df['Plant_Health_Status'])
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# Features and target
features = ['Soil_Moisture','Ambient_Temperature','Soil_Temperature',
            'Humidity','Light_Intensity','Soil_pH','Nitrogen_Level',
            'Phosphorus_Level','Potassium_Level','Chlorophyll_Content',
            'Electrochemical_Signal']

X = df[features]
y = df['Plant_Health_Status_Encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model, scaler, and label encoder
joblib.dump(model, "plant_ai_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model, scaler, and label encoder saved. Ready for Backend!")
