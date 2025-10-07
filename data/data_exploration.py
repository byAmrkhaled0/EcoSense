import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/plant_health_data.csv")

# Display column names
print("🧾 Column Names:")
print(df.columns)

# Display first 5 rows
print("📊 First 5 rows:")
print(df.head())

# Info about columns
print("ℹ️ Data Types and Non-Null Counts:")
print(df.info())

# Statistical summary for numeric columns
print("📈 Numeric Summary:")
print(df.describe())

# Pairplot with Plant Health Status
sns.pairplot(df, hue="Plant_Health_Status")
plt.show()

# Histogram for all numeric features
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.show()
