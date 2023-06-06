import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

def perform_chi_square_test(df, column_name, injury_column):
    # Create the contingency table
    contingency_table = pd.crosstab(df[gender_column], df[injury_column])
    
    # Perform the chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Print the results
    print("Chi-square test statistic:", chi2)
    print("P-value:", p_value)
    print("Degrees of freedom:", dof)
    
    


def plot_contingency_table(df, column1, column2):
    # Create the contingency table
    contingency_table = pd.crosstab(df[column1], df[column2])

    # Plot the contingency table as a bar chart
    contingency_table.plot(kind='bar', figsize=(20, 20))
    plt.show()

