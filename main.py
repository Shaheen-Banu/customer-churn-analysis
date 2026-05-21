# ==============================
# CUSTOMER CHURN ANALYSIS PROJECT
# ==============================

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("🚀 Customer Churn Project Started")

# STEP 2: Load Dataset
df = pd.read_csv("churn.csv")

print("\n📊 First 5 Rows:")
print(df.head())

# STEP 3: Understand Data
print("\n📐 Shape of Data:", df.shape)
print("\n📌 Columns:", df.columns)

print("\n🔍 Data Info:")
print(df.info())

# STEP 4: Check Missing Values
print("\n❗ Missing Values:")
print(df.isnull().sum())

# ==============================
# STEP 5: DATA CLEANING
# ==============================

print("\n🧹 Cleaning Data...")

# Remove unnecessary column
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Drop missing values
df = df.dropna()

# Convert Churn to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\n✅ After Cleaning:")
print(df.info())

# ==============================
# STEP 6: EXPLORATORY DATA ANALYSIS
# ==============================

print("\n📊 Generating Visualizations...")

# Churn Distribution
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Tenure vs Churn
plt.figure()
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()

# Monthly Charges vs Churn
plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Contract Type vs Churn
if "Contract" in df.columns:
    plt.figure()
    sns.countplot(x="Contract", hue="Churn", data=df)
    plt.title("Contract Type vs Churn")
    plt.xticks(rotation=30)
    plt.show()

# ==============================
# STEP 7: PREPROCESSING
# ==============================

print("\n⚙️ Preparing Data for Model...")

# Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert categorical to numeric
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# STEP 8: MODEL TRAINING
# ==============================

print("\n🤖 Training Model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# STEP 9: PREDICTION
# ==============================

y_pred = model.predict(X_test)

# ==============================
# STEP 10: EVALUATION
# ==============================

print("\n📊 Model Evaluation:")

accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", accuracy)

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# STEP 11: FINAL INSIGHTS
# ==============================

print("\n💡 Key Insights:")
print("- Customers with low tenure are more likely to churn")
print("- High monthly charges increase churn risk")
print("- Monthly contract users churn more than long-term users")

print("\n🎯 Project Completed Successfully!")