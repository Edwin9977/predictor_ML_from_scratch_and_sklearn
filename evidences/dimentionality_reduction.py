import pandas as pd
from sklearn.linear_model import Perceptron

dataset = pd.read_csv("Social_Network_Ads.csv")

# checar problema de reduccion de dimencionalidad, dimentionality reduction, fixture selection, random forest, etc.
# podemos generar mas columnas para cada genero en vez de agregarle valores a cada genero.
dataset = dataset.drop(columns = 'User ID')

dataset["Male"] = ' ' #(dataset["Gender"] == "Male").astype(int)
dataset['Female'] = ' '

for i in range(len(dataset)):
    if (dataset.iloc[i, 0] == 'Male'):
        dataset.iloc[i, 4] = 1
        dataset.iloc[i, 5] = 0
    else:
        dataset.iloc[i, 5] = 1
        dataset.iloc[i, 4] = 0


dataset = dataset.drop(columns = 'Gender')
# print(dataset.columns)

dataset = dataset.reindex(['Age', 'EstimatedSalary', 'Male', 'Female', 'Purchased'], axis=1)
x = dataset.reindex(['Age', 'EstimatedSalary', 'Male', 'Female'], axis=1)
y = dataset.reindex(['Purchased'], axis=1)
x_train = x[:319]
y_train = y[:319].values.ravel()
x_val = x[320:]
y_val = y[320:].values.ravel()
# perceptron
# y es para Purchased
# los labels se llaman Y
# los otros features se llaman X

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_train, y_train)
Perceptron()
print(clf.score(x_train, y_train))
print(clf.score(x_val, y_val))
# modelo predictivo genera una funcion