# Without Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Total number of values
n = len(X)


#finding m and c
numer = 0
denom = 0
for i in range(n):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = numer/denom
c = mean_y - (m*mean_x)

# Printing Coefficients
print("Slope(Coefficient) - m = ", m)
print("Intercept - c = ", c)


# Plotting Values and Regression Line
max_x = np.max(X) + 50
min_x = np.min(X) - 50

# Calculating line values x and y
x = np.linspace(min_x, max_x, 500)
y = c + m * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Traning Data')

plt.xlabel('X-Input')
plt.ylabel('Y-Label')
plt.legend()
plt.show()


#plotting with test data
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 500)
y = c + m * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X_test, Y_test, c='#003F72', label='Test Data')

plt.xlabel('X-Input')
plt.ylabel('Y-Predict')
plt.legend()
plt.show()


#Calculating Error
error = 0
n = len(X_test)
for i in range(n):
    pred = m * X_test[i] + c
    y = Y_test[i]
    error += (pred - y)**2

print("Sum of squares error on test data is: ", error)
mse = (error/n)
print("MSE: ", mse)
rmse = np.sqrt(error/n)
print("RMSE: ", rmse)


# Calculating R^2 Score
ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = c + m * X_test[i]
    ss_tot += (Y_test[i] - mean_y) ** 2
    ss_res += (Y_test[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R^2 Score: ", r2)


# #Computing data
# X = data['TV'].values[:101]
# X = data['Radio'].values[:101]
# X = data['Newspaper'].values[:101]
# Y = data['Sales'].values[:101]
# X_test = data['TV'].values[101:]
# X_test = data['Radio'].values[101:]
# X_test = data['Newspaper'].values[101:]
# Y_test = data['Sales'].values[101:]
# mean_x = np.mean(X)
# mean_y = np.mean(Y)

# # Total number of values
# n = len(X)