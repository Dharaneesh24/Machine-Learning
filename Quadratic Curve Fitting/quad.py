import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scipy.stats as stats
# from sklearn.metrics import r2_score

dataset = pd.read_csv('ball.csv')
print(dataset.shape)
print(dataset.head())

#Computing data
X = dataset['time']
Y = dataset['height']

#create scatterplot
plt.scatter(X, Y)

#polynomial fit with degree = 2
model = np.poly1d(np.polyfit(X, Y, 2))

#add fitted polynomial line to scatterplot
polyline = np.linspace(1, 60, 50)
plt.scatter(X, Y)
plt.plot(polyline, model(polyline))
plt.show()

print(model)

#define function to calculate r-squared
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    print(coeffs)
    p = np.poly1d(coeffs)

    #Calculating Error
    error = 0
    n = len(X)
    for i in range(n):
        pred = (coeffs[0] * (X[i] * X[i])) + (coeffs[1] * X[i]) + coeffs[2]
        y1 = Y[i]
        error += (pred - y1)**2
    print("Sum of squares error on test data is: ", error)
    mse = (error/n)
    print("MSE: ", mse)
    rmse = np.sqrt(error/n)
    print("RMSE: ", rmse)

    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = ssreg / sstot

    return results


#find r-squared of polynomial model with degree = 2
print(polyfit(X, Y, 2))