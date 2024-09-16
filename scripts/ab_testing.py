import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

# 1. Data Loading and Preprocessing
def load_data(filepath):
    """
    Load the dataset from the file path.
    """
    data = pd.read_csv(filepath)
    return data

# 2. Select Metrics (KPI)
def calculate_kpis(data):
    """
    Add new KPI columns such as Profit Margin to the dataset.
    """
    data['ProfitMargin'] = data['TotalPremium'] - data['TotalClaims']
    return data

# 3. Data Segmentation (A/B Testing Groups)
def segment_data(data, feature_column, value_A, value_B):
    """
    Segment the data into Group A (Control) and Group B (Test) based on the feature.
    """
    group_A = data[data[feature_column] == value_A]
    group_B = data[data[feature_column] == value_B]
    return group_A, group_B

# 4. Statistical Testing Functions
def chi_square_test(group_A, group_B, feature_column):
    """
    Perform a Chi-Square test for categorical features between Group A and Group B.
    """
    contingency_table = pd.crosstab(group_A[feature_column], group_B[feature_column])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return p

def t_test(group_A, group_B, numerical_column):
    """
    Perform a t-test for numerical features between Group A and Group B.
    """
    t_stat, p = stats.ttest_ind(group_A[numerical_column], group_B[numerical_column], nan_policy='omit')
    return p

def z_test(group_A, group_B, numerical_column):
    """
    Perform a z-test for large sample numerical features between Group A and Group B.
    """
    mean_A = group_A[numerical_column].mean()
    mean_B = group_B[numerical_column].mean()
    std_A = group_A[numerical_column].std()
    std_B = group_B[numerical_column].std()
    
    n_A = len(group_A)
    n_B = len(group_B)
    
    z_score = (mean_A - mean_B) / np.sqrt((std_A**2 / n_A) + (std_B**2 / n_B))
    p_value = stats.norm.sf(abs(z_score)) * 2  # two-tailed test
    return p_value

# 5. Analyze and Report
def analyze_results(p_value, alpha=0.05):
    """
    Analyze the p-value to accept or reject the null hypothesis.
    """
    if p_value < alpha:
        return "Reject the Null Hypothesis: Significant difference found."
    else:
        return "Fail to Reject the Null Hypothesis: No significant difference."

# 6. Full A/B Testing Pipeline
def ab_testing_pipeline(data, feature_column, value_A, value_B, kpi_column, test_type='t_test'):
    """
    Run the full A/B testing pipeline for a given feature.
    """
    # Segment the data
    group_A, group_B = segment_data(data, feature_column, value_A, value_B)
    
    # Conduct statistical test
    if test_type == 'chi_square':
        p_value = chi_square_test(group_A, group_B, feature_column)
    elif test_type == 't_test':
        p_value = t_test(group_A, group_B, kpi_column)
    elif test_type == 'z_test':
        p_value = z_test(group_A, group_B, kpi_column)
    
    # Analyze the results
    result = analyze_results(p_value)
    return result, p_value
