## 📊 Dataset  
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
- **File used:** `train.csv`  
- **Size:** 891 rows × 12 columns  

## ⚙️ Project Workflow  

1. **Import libraries and dataset**  
2. **Load dataset**
3. **Inspect dataset** – shape, head, info, null values, duplicates  
4. **Data Cleaning:**     - Handle missing values (`Age`, `Cabin`, `Embarked`)  
   - Fix data types (categorical conversion)  
   - Remove duplicates  
   - Feature engineering (extract `Title` from `Name`)  
5. **Exploratory Data Analysis (EDA):**  
   - Univariate analysis (Age, Fare, Survival distribution)  
   - Bivariate analysis (Survival vs Gender, Survival vs Pclass)  
   - Correlation heatmap  
   - Visualizations with Seaborn & Matplotlib  
6. **Insights**


# Titanic Data Cleaning & EDA Project

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv("train.csv")
print("Shape:", df.shape)
print(df.head())

# 3. Data Inspection
print(df.info())
print(df.isnull().sum())

# 4. Data Cleaning
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # too many missing

# Fix data types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Feature Engineering: Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 5. Exploratory Data Analysis
# Univariate Analysis
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Survived')
plt.title("Survival Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Bivariate Analysis
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival by Passenger Class")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 6. Insights
print("Key Findings:")
print("- Females had higher survival rates than males.")
print("- Passengers in 1st class had higher chances of survival.")
print("- Younger passengers had slightly better survival rates.")
print("- Family size (SibSp + Parch) impacted survival chances.")
