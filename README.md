# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SAFIRA BARVEEN S
RegisterNumber: 212224230235
*/
```
```.py
import pandas as pd
data=pd.read_csv("Salary.csv")
df=pd.DataFrame(data)
df.head()
from sklearn.preprocessing  import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
x=df[["Position","Level"]]
y=df["Salary"]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


mae=mean_absolute_error(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
rmse=np.sqrt(mse)
print("MAE:", mae)
print("MSE:", mse)
print("R² score:", r2)
print("RMSE:",rmse)

dt.predict([[4,5]])
```

## Output:
### Data Head:
<img width="300" height="207" alt="image" src="https://github.com/user-attachments/assets/5939e52c-411e-443e-ba5a-7397d029435e" />

### Data head for salary:
<img width="299" height="212" alt="image" src="https://github.com/user-attachments/assets/28847ef0-7e10-4ca2-a39e-fcc029248f8e" />

### Mean Squared Error:
<img width="199" height="28" alt="image" src="https://github.com/user-attachments/assets/a5e0f441-2f5e-4da6-8b56-734090aa9ef5" />

### Mean absolute error:
<img width="154" height="23" alt="image" src="https://github.com/user-attachments/assets/d353f5c4-6f56-4a8c-8a5e-534946946be6" />

### R2_score:
<img width="304" height="21" alt="image" src="https://github.com/user-attachments/assets/7bf511e9-8b4c-4383-afda-9286e89dabe1" />

### Root mean squared error:
<img width="266" height="23" alt="image" src="https://github.com/user-attachments/assets/9790aa0a-8fdf-4b66-8777-e105921965ce" />

### Predicted value:
<img width="1251" height="121" alt="image" src="https://github.com/user-attachments/assets/5d382312-3fa0-4839-bd5e-a0e419507e69" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
