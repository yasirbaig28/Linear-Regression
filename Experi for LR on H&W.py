#Expeiment of LinerRegression on Heights and Weight csv file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Heights_and_Weights.csv')
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
y_pred1 = regressor.predict([[1.62]])
print(y_pred1)

check=pd.DataFrame(y_test,y_pred)
print(check)

plt = plt
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('HEIGHTS VS WEIGHTS')
plt.xlabel('HEIGHT')
plt.ylabel('WEIGHT')
plt.show()

