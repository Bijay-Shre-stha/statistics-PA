import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read the data from CSV
df = pd.read_csv('lab.csv')

# Separate independent variables (X1, X2, X3) and dependent variable (Y)
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

# Get the coefficients (slope) and intercept
coefficients = model.coef_
intercept = model.intercept_

# Make predictions for the scatter plot
predictions = model.predict(X)

# Calculate R^2 value
r2_value = r2_score(Y, predictions)

# Scatter plot
plt.scatter(Y, predictions)
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], linestyle='--', color='red', label='Perfect Fit')
plt.title('Scatter Plot with Regression Line')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')

# Display regression equation on the plot
equation_text = f'Regression Equation:\nY = {intercept:.2f} + {coefficients[0]:.2f}*X1 + {coefficients[1]:.2f}*X2 + {coefficients[2]:.2f}*X3'
plt.text(min(Y), max(predictions), equation_text, fontsize=10, verticalalignment='bottom', horizontalalignment='left')

# Display R^2 value on the plot
r2_text = f'R^2 Value: {r2_value:.2f}'
plt.text(min(Y), max(predictions)-2, r2_text, fontsize=10, verticalalignment='bottom', horizontalalignment='left')

# Show the plot
plt.show()
