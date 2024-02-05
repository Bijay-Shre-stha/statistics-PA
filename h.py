import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Read the data from CSV
df = pd.read_csv('lab.csv')

# Separate independent variables (X1, X2, X3) and dependent variable (Y)
X = df[['X1', 'X2', 'X3']]

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF values
print("VIF Values:")
print(vif_data)
