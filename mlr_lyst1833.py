import pandas as pd
import numpy as np

datasets=pd.read_csv("50_Startups.csv")
#print(datasets)

X=datasets.iloc[:,:-1].values
Y=datasets.iloc[:,4].values
#print(X)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A=make_column_transformer((OneHotEncoder(categories="auto"),[3]),remainder="passthrough")

X=A.fit_transform(X)
#print(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.20,random_state=0)




from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

y_pred= regressor.predict(X_test)
#print(X_test)
#print(pd.DataFrame(Y_test,y_pred))
y_pred1=regressor.predict([[1.0,0.0,0.0,200000,10000,30000]])
print(y_pred1)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test,y_pred))

print(regressor.score(X_test,Y_test))






