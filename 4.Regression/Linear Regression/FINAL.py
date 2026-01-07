# =========================================
# MULTIPLE LINEAR REGRESSION (FINAL CODE)
# =========================================

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================================
# 2. Load Dataset
# =========================================
df = pd.read_csv("dataset.csv")

# =========================================
# 3. Basic Data Checking
# =========================================
print(df.head())
print(df.info())
print(df.describe())

# =========================================
# 4. Data Cleaning
# =========================================

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Handle missing values
# Numerical columns -> Median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns -> Mode
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# # Outlier removal using IQR method
# for col in num_cols:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1

#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR

#     df = df[(df[col] >= lower) & (df[col] <= upper)]

# =========================================
# 5. Encode Categorical Data (Label Encoding)
# =========================================
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# =========================================
# 6. Feature and Target Separation
# =========================================
X = df.drop('Target', axis=1)   # Independent variables
y = df['Target']                # Dependent variable

# =========================================
# 7. Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 8. Feature Scaling (Only Numerical Data)
# =========================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================
# 9. Train Multiple Linear Regression Model
# =========================================
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# =========================================
# 10. Prediction
# =========================================
y_pred = mlr.predict(X_test)

# =========================================
# 11. Model Evaluation
# =========================================
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)
print("Model Accuracy:", r2 * 100, "%")

# =========================================
# 12. ACTUAL vs PREDICTED GRAPH
# =========================================

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs Predicted (Best Fit Line)")
plt.show()

# # =========================================
# # 13. Predicting from model trained
# # =========================================

# Random_input = pd.DataFrame([[
#     120000,   # R&D Spend
#     50000,    # Administration
#     30000,    # Marketing Spend
#     1         # State (encoded value)
# ]], columns=X.columns)

# Random_input_scaled = scaler.transform(Random_input)
# y_pred = mlr.predict(Random_input_scaled)
# print("Predicted Output:", y_pred[0])