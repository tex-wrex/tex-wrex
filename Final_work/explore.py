import pandas as pd
from scipy.stats import chi2_contingency


def perform_chi_square_test(df, column_name, injury_column):
    # Create the contingency table
    contingency_table = pd.crosstab(df[gender_column], df[injury_column])
    
    # Perform the chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Print the results
    print("Chi-square test statistic:", chi2)
    print("P-value:", p_value)
    print("Degrees of freedom:", dof)
