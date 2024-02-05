import pandas as pd
import statsmodels.api as sm

# Read the data from CSV
df = pd.read_csv('lab.csv')

# Separate independent variables (X1, X2, X3) and dependent variable (Y)
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Create a linear regression model using statsmodels
model = sm.OLS(Y, X).fit()

# Print the summary to get the overall regression significance test
print(model.summary())
