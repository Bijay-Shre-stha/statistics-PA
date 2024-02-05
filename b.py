import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the data from CSV
df = pd.read_csv(filepath_or_buffer='lab.csv')

# Separate independent variables (X1, X2, X3) and dependent variable (Y)
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

# Given values for prediction
new_data = pd.DataFrame({'X1': [1], 'X2': [14], 'X3': [25]})

# Make predictions
predicted_y = model.predict(new_data)

# Print the estimated value of Y
print(f'Estimated Y for X1=1, X2=14, X3=25 is: {predicted_y[0]:.2f}')
