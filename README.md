# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date: 19.08.2025
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
A - LINEAR TREND ESTIMATION

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("C:\\Users\\admin\\time series\\housing_price_dataset.csv")

# Group by YearBuilt to create a time series of average prices per year
price_series = data.groupby("YearBuilt")["Price"].mean().reset_index()

# Linear Regression (trend)
X = price_series[["YearBuilt"]]
y = price_series["Price"]

linear_model = LinearRegression()
linear_model.fit(X, y)
price_series["linear_trend"] = linear_model.predict(X)

# Polynomial Regression (Quadratic)
poly_degree = 2
X_poly = np.column_stack([X.values**i for i in range(1, poly_degree + 1)])
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
price_series["poly_trend"] = poly_model.predict(X_poly)

# ---- Plot Linear Trend ----
plt.figure(figsize=(12, 6))
plt.plot(price_series["YearBuilt"], price_series["Price"], 
         label="Original (Avg Price per Year)", color="blue")
plt.plot(price_series["YearBuilt"], price_series["linear_trend"], 
         label="Linear Trend", color="red", linestyle="--", linewidth=2)
plt.title("Linear Trend Estimation")
plt.xlabel("Year Built")
plt.ylabel("Average Price")
plt.legend()
plt.show()

# ---- Plot Polynomial Trend ----
plt.figure(figsize=(12, 6))
plt.plot(price_series["YearBuilt"], price_series["Price"], 
         label="Original (Avg Price per Year)", color="blue")
plt.plot(price_series["YearBuilt"], price_series["poly_trend"], 
         label="Polynomial Trend (Quadratic)", color="green", linestyle="-", linewidth=2)
plt.title("Polynomial Trend Estimation (Quadratic)")
plt.xlabel("Year Built")
plt.ylabel("Average Price")
plt.legend()
plt.show()

# Print coefficients to check overlap
print("Linear coefficients:", linear_model.coef_, "Intercept:", linear_model.intercept_)
print("Polynomial coefficients:", poly_model.coef_, "Intercept:", poly_model.intercept_)
```

### OUTPUT
A - LINEAR TREND ESTIMATION

<img width="1241" height="648" alt="image" src="https://github.com/user-attachments/assets/271d6722-9a40-4ad7-a428-e4249e43415e" />

B- POLYNOMIAL TREND ESTIMATION


<img width="1280" height="648" alt="image" src="https://github.com/user-attachments/assets/8939201a-1a44-40c6-af4c-e2886138e02a" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
