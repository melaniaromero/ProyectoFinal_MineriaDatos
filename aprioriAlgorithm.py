import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar el archivo CSV
file_path = 'bank_data.csv'
# Guardarlo en bank_data
bank_data = pd.read_csv(file_path)

#Escribir las columnas que vamos a binarizar.
columns_to_binarize = [
    'STATUS_SOLICITUD', 'TIPO_CTE', 'APROBACION_TC', 'TIPO_VIVIENDA', 'ESCOLARIDAD',
    'NIVEL_RIESGO', 'COMPROBANTE_INGRESOS', 'SEGMENTO_CLIENTE','CAPACIDAD_PAGO_TOTAL'
]
#Aplicar la binarización 
bank_data_full_binarized = pd.get_dummies(bank_data[columns_to_binarize])
# Guardar el csv transformado
bank_data_full_binarized.to_csv('bank_data_full_binarized.csv', index=False)

#Apriori
min_support = 0.6  #Soporte mínimo
min_confidence = 0.7  #Confianza mínima

#Encontrar los conjuntos de ítems frecuentes
conjuntos_frecuentes = apriori(bank_data_full_binarized, min_support=min_support, use_colnames=True)

#Generar reglas de asociación
reglas = association_rules(conjuntos_frecuentes, metric="confidence", min_threshold=min_confidence, num_itemsets=4200)

# Guardar conjuntos frecuentes y reglas de asociación en un txt.
with open('association_results.txt', 'w') as file:
    # Conjuntos frecuentes
    file.write("Conjuntos frecuentes encontrados:\n\n")
    for idx, row in conjuntos_frecuentes.iterrows():
        file.write(f"Conjunto {idx}: {set(row['itemsets'])}\n")
        file.write(f"Soporte: {row['support']:.4f}\n\n")

    # Reglas de asociación
    file.write("\nReglas de asociación generadas:\n\n")
    for idx, row in reglas.iterrows():
        file.write(f"Regla {idx}:\n")
        file.write(f"  Antecedente: {set(row['antecedents'])}\n")
        file.write(f"  Consecuente: {set(row['consequents'])}\n")
        file.write(f"  Confianza: {row['confidence']:.4f}\n")
        file.write(f"  Soporte: {row['support']:.4f}\n")
        file.write(f"  Lift: {row['lift']:.4f}\n\n")

