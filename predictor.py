import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("Indicadores_municipales_sabana_DA.csv", encoding="latin_1")

# columns to remove
columns_to_drop = ["ent", "nom_ent", "mun", "clave_mun", "nom_mun"]
for i in range(len(columns_to_drop)):
    dataset = dataset.drop(columns = columns_to_drop[i])

# columns that don't contain number values
gdo_rezsoc = ['gdo_rezsoc00', 'gdo_rezsoc05', 'gdo_rezsoc10']
for column in dataset.columns:
    if column not in gdo_rezsoc:
        # calculating the mean
        column_mean = dataset[column].mean()
        if dataset[column].isnull().values.any():
            # if value is null, then replace it
            dataset[column].fillna(column_mean, inplace=True)
    else:
        if dataset[column].isnull().values.any():
            # if any non numerical value is null. then replace it with the word Bajo
            dataset[column].fillna('Bajo', inplace=True)

# unique values
valores_unicos_en_gdo = []
for gdo in gdo_rezsoc:
    valores_unicos_en_gdo.append(dataset[gdo].unique())

# new column names
new_columns = []
for i in range(len(gdo_rezsoc)):
    new_columns.append([])
    for j in range(len(valores_unicos_en_gdo[i])):
        new_columns[i].append(f"{gdo_rezsoc[i]}_{valores_unicos_en_gdo[i][j]}")

# here we add the new columns with its corresponding values
for i in range(len(new_columns)):
    for j in range(len(new_columns[i])):
        dataset[new_columns[i][j]] = (dataset[gdo_rezsoc[i]] == valores_unicos_en_gdo[i][j]).astype(int)

# remove columns with no numerical values
for i in range(len(gdo_rezsoc)):
    dataset = dataset.drop(columns = gdo_rezsoc[i])

# columna que usaremos como Y
column_to_predict = 'pobreza'  

# verificamos si la columna existe en el DataFrame
if column_to_predict in dataset.columns:
    # Eliminar la columna del DataFrame y la guarda en una variable
    column = dataset.pop(column_to_predict)
    # Agregar la columna al final del DataFrame
    dataset[column_to_predict] = column
else:
    print(f"La columna {column_to_predict} no existe en el DataFrame.")

# Calcula la desviación estándar para cada columna
std_dev = dataset.std()

# Define un umbral (por ejemplo, 3 veces la desviación estándar)
threshold = 3

# Identifica los índices de las filas que contienen outliers
outliers = (np.abs(dataset - dataset.mean()) < threshold * dataset.std()).all(axis=1)

# Elimina las filas que contienen outliers del DataFrame
dataset = dataset[~outliers]

total_rows = len(dataset)
eigthy = int(round(total_rows*0.8, 0))

y_dataset = dataset[column_to_predict]
x_dataset = dataset.drop(columns=column_to_predict)

x_train = x_dataset[:eigthy].values
y_train = y_dataset[:eigthy].values

x_val = x_dataset[eigthy:].values
y_val = y_dataset[eigthy:].values
# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# descenso de gradiente
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        theta = theta - (alpha/m) * X.T.dot(errors)
    return theta

# Función para calcular el error cuadrático medio normalizado (NMSE)
def normalized_mean_squared_error(y, y_pred):
    y_range = np.max(y) - np.min(y)
    return np.mean((y - y_pred) ** 2) / (y_range**2)

# para normalizar las características
def normalize_features(X):
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_normalized

# para normalizar la variable objetivo
def normalize_target(y):
    y_normalized = (y - y.mean()) / y.std()
    return y_normalized

# Agregar una columna de unos para el término de sesgo
x_train_with_bias = np.c_[np.ones((x_train.shape[0], 1)), normalize_features(x_train)]
y_train_normalized = normalize_target(y_train)

# Inicialización de theta con valores aleatorios
initial_theta = np.random.rand(x_train_with_bias.shape[1])

# Tasa de aprendizaje y número de iteraciones
learning_rate = 0.001
iterations = 1000

# Realizar el descenso de gradiente
theta = gradient_descent(x_train_with_bias, y_train_normalized, initial_theta, learning_rate, iterations)

# Agregar una columna de unos para el término de sesgo en el conjunto de validación
x_val_with_bias = np.c_[np.ones((x_val.shape[0], 1)), normalize_features(x_val)]
y_val_normalized = normalize_target(y_val)

# predicciones
y_val_pred = x_val_with_bias.dot(theta)

# normalizacion de mse
nmse = normalized_mean_squared_error(y_val_normalized, y_val_pred)
print(f"Error cuadrático medio normalizado en el conjunto de validación: {nmse}")


# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
# libreria
# Crear un objeto de regresión lineal
model = LinearRegression()

# Entrenamiento
model.fit(x_train, y_train)

# predicciones
y_val_pred = model.predict(x_val)

# Calcular el error cuadrático medio (MSE) en el conjunto de datos de validación
mse = mean_squared_error(y_val, y_val_pred)

print(f"Error cuadrático medio en el conjunto de datos de validación (library): {mse}")