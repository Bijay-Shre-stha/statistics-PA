import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

# Read the data from CSV
df = pd.read_csv('lab.csv')

# Separate independent variables (X1, X2, X3) and dependent variable (Y)
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Create a linear regression model using statsmodels
model = sm.OLS(Y, X).fit()

# Get residuals
residuals = model.resid

# Q-Q plot for residuals
sm.qqplot(residuals, line='45', fit=True)

# Add title and labels
plt.title('Normal Probability Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

# Display the plot
plt.show()
