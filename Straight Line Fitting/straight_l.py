# # polyfit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Reading Data
data = pd.read_csv('simple_linear_regression.csv')
print(data.shape)
print(data.head())

#Computing data
X = data['X'].values[:150]
Y = data['Y'].values[:150]
X_test = data['X'].values[150:]
Y_test = data['Y'].values[150:]
mean_x = np.mean(X)
mean_y = np.mean(Y)
plt.scatter(X, Y)

# model = np.polyfit(X, Y, 1)
# # This executes the polyfit method from the numpy library 
# # that we have imported before. 
# # It needs three parameters: 
# # the previously defined input and output variables (X, Y)
# # and an integer, too: 1. 
# # This latter number defines the degree of the polynomial you want to fit.

# print(model)
# # ouputs array([m, c])
# # m - regression coefficient
# # c - intercept

# predict = np.poly1d(model)

# x_input = 301
# print(predict(x_input))

# # To calculate the accuracy of your model. 
# # The R-squared (R2) value.
# print(r2_score(Y, predict(X)))

# x_lin_reg = range(0, 301)
# y_lin_reg = predict(x_lin_reg)
# plt.scatter(X, Y)
# plt.plot(x_lin_reg, y_lin_reg, c = 'r')
# plt.show()


# sklearn
from sklearn.linear_model import LinearRegression
npx = np.array(X).reshape((-1, 1))
model = LinearRegression().fit(npx,Y)
print(f"slope: {model.coef_[0]}")
print(f"intercept: {model.intercept_}")

#Calculating error
error = 0
n = len(X_test)
for i in range(n):
    pred = model.predict(np.array(X_test[i]).reshape(-1,1))
    y = Y_test[i]
    error+= (pred - y)**2

print("Sum of squares error on test data is: ", error)
mse = (error/n)
print("MSE: ", mse)
rmse = np.sqrt(error/n)
print("RMSE: ", rmse)