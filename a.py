import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the data from CSV
df = pd.read_csv('lab.csv')

# Separate independent variables (X1, X2, X3) and dependent variable (Y)
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

# Given values for prediction
new_data = pd.DataFrame({'X1': [1], 'X2': [14], 'X3': [25]})

# Get the coefficients (slope) and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the regression equation
print(f'Regression Equation: Y = {intercept:.2f} + {coefficients[0]:.2f}*X1 + {coefficients[1]:.2f}*X2 + {coefficients[2]:.2f}*X3')
