import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Load dataset from CSV
df = pd.read_csv('Position_Salaries.csv') 

# Extract features and target variable
X = df[['Feature']] 
y = df['Target'] 

def svr(X, y, kernel, label):
    # Fit SVR model
    svr_model = SVR(kernel=kernel, C=100, gamma=2, epsilon=1)
    svr_model.fit(X, y)

    # Predict
    X_test = np.arange(X.min().min(), X.max().max(), 0.01)[:, np.newaxis]
    y_pred = svr_model.predict(X_test)

    # Plot the results
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X_test, y_pred, lw=2, label=label)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

# Call the svr function for Linear kernel
svr(X, y, 'linear', 'SVR (Linear kernel)')

# Call the svr function for Polynomial kernel
svr(X, y, 'poly', 'SVR (Polynomial kernel)')

# Call the svr function for RBF kernel
svr(X, y, 'rbf', 'SVR (RBF kernel)')
