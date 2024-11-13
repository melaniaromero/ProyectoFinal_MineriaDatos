import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

#Transformar los valores de la columna 'CAPACIDAD_PAGO_TOTAL'
#Si es igual a 0 la capacidad de pago es Baja
#Si es mayor a 0 y menor o igual a 1 la capacidad de pago es Aceptable
#Si es mayor es 1 la capacidad de pago es Alta

def transformar_capacidad_pago(valor):
    if valor == 0:
        return 'BAJO'
    elif 0 < valor <= 1:
        return 'ACEPTABLE'
    elif valor > 1:
        return 'ALTO'

# Cargar el archivo CSV
file_path = 'bank_data.csv'
bank_data = pd.read_csv(file_path)

#Aplicar transformación a 'CAPACIDAD_PAGO_TOTAL'
bank_data['CAPACIDAD_PAGO_TOTAL'] = bank_data['CAPACIDAD_PAGO_TOTAL'].apply(transformar_capacidad_pago)
#Guardar en bank_data_CAPACIDAD.csv
bank_data.to_csv('bank_data_CAPACIDAD.csv', index=False)
# Un comentario sobre las líneas que puedes usar para el analisis del arbol
 #   'STATUS_SOLICITUD', 'TIPO_CTE', 'APROBACION_TC', 'TIPO_VIVIENDA', 'ESCOLARIDAD',
 #  'NIVEL_RIESGO', 'COMPROBANTE_INGRESOS', 'SEGMENTO_CLIENTE','CAPACIDAD_PAGO_TOTAL'

# Columnas de características (X) y la columna objetivo (y)
X = pd.get_dummies(bank_data[['STATUS_SOLICITUD', 'CAPACIDAD_PAGO_TOTAL', 'APROBACION_TC', 'NIVEL_RIESGO']], drop_first= True) 
y = bank_data['TIPO_CTE']  # Variable objetivo

#Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Crear el modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

#Entrenar el modelo
clf.fit(X_train, y_train)

#Hacer predicciones sobre el conjunto de prueba
y_pred = clf.predict(X_test)

#Evaluar el rendimiento del modelo
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):}")
print("************* Reporte de clasificación *************")
print(classification_report(y_test, y_pred))

# Visualizar el árbol de decisión
plt.figure(figsize=(17,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_, rounded=True)
plt.show()
