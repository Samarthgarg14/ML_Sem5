import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("/Users/samarthgarg/Downloads/ML ETP/DataSets/Position_Salaries.csv")

# Features and target
X = df[['Independent variable']]
y = df['Dependent variable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Polynomial features
poly = PolynomialFeatures(degree=3)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

# Model training
pr = LinearRegression()
pr.fit(X_train, y_train)

# Prediction & evaluation
y_pred = pr.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_scaled = sc.transform(X_range)
X_range_poly = poly.transform(X_range_scaled)
y_range_pred = pr.predict(X_range_poly)

plt.scatter(X, y)
plt.plot(X_range, y_range_pred)
plt.title("Polynomial Regression (Degree = 3)")
plt.show()

# # Manual prediction
# manual_input = np.array([[6.5]])
# manual_input_scaled = sc.transform(manual_input)
# manual_input_poly = poly.transform(manual_input_scaled)
# predicted_salary = pr.predict(manual_input_poly)

# print("Predicted Salary for Level 6.5:", predicted_salary[0])
