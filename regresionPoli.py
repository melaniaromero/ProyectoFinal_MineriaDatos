import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Cargar los datos
data = pd.read_csv('bank_data1.csv')

# Seleccionar las columnas necesarias
X = data[['CAPACIDAD_PAGO_TOTAL']].values
y = data['SOLICITUDES_RECHAZADAS'].values

# Transformación polinomial de grado 5
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

# Ajustar el modelo de regresión lineal a los datos transformados
model = LinearRegression()
model.fit(X_poly, y)

# Crear valores de predicción
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = model.predict(X_range_poly)

# Graficar los datos y la curva de regresión
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Datos originales')
plt.plot(X_range, y_pred, color='red', label='Regresión polinomial de grado 5')
plt.xlabel('CAPACIDAD_PAGO_TOTAL')
plt.ylabel('SOLICITUDES_RECHAZADAS')
plt.title('CAPACIDAD_PAGO_TOTAL vs SOLICITUDES_RECHAZADAS')
plt.legend()
plt.show()
