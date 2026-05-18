# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection and Preparation: Load the Salary dataset containing Position, Level, and Salary columns. Select Level as the independent variable (X) and Salary as the dependent variable (y).
2. Model Training: Initialize the Decision Tree Regressor with a defined random state and fit the model using the Level and Salary data to learn the salary patterns at each level.
3. Prediction: Create a fine-grained range of Level values using np.arange for smooth curve plotting and predict the salary for each value. Also predict the salary for a specific level (e.g., Level 6.5) to validate the model.
4. Visualization: Plot the actual salary data points as red scatter dots and overlay the predicted salary as a blue step-curve line. Label the axes, add a title and legend to interpret the Decision Tree Regression model output clearly.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SUDESAN T 
RegisterNumber: 212225240161
*/
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

uploaded = files.upload()

df = pd.read_csv('Salary.csv')

X = df[['Level']]
y = df['Salary']

model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

X_grid = pd.DataFrame(np.arange(min(X['Level']), max(X['Level']) + 0.01, 0.01), columns=['Level'])
y_pred_grid = model.predict(X_grid)

print("Predicted Salary for Level 6.5:", model.predict(pd.DataFrame([[6.5]], columns=['Level']))[0])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', s=80, zorder=5, label='Actual Salary')
plt.plot(X_grid, y_pred_grid, color='blue', linewidth=2, label='Predicted Salary')
plt.xlabel('Level', fontsize=13)
plt.ylabel('Salary', fontsize=13)
plt.title('Decision Tree Regression Model', fontsize=15)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

## Output:

<img width="991" height="637" alt="{1CBEE776-EBCD-4D4A-A6D9-32EB45AB9889}" src="https://github.com/user-attachments/assets/1abb7943-517e-4371-accf-f8b211fb438a" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
