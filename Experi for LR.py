# To Read the CSV file and predict the future results
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('test.csv')
# x is the X coulmn
x = dataset.iloc[:,:-1].values #In multiple column to acces all other columns execpt y column i.e (-1)
# values because to acces only the values not the name of the column

# y column
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.05,random_state=22)

# data=pd.DataFrame(x_train,y_train)
# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression 

regressor=LinearRegression() # naming LinearRegression as regressor
regressor.fit(x_train,y_train) # fit is to tell the algorithm   that which is x and y data

# Predicting the Test set results
y_pred = regressor.predict(x_test)

print(x_test)
crosscheckdata = pd.DataFrame(y_test,y_pred)
print(crosscheckdata)

y_pred1 = regressor.predict([[5]])

print(y_pred1) 
print(regressor.intercept_)
print(regressor.coef_)

#plotting dataset
plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train))
plt.title("DATA GRAPH")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()
