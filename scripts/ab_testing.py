import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import ztest

def chi_squared_test(group1, group2, feature):
    contingency_table = pd.crosstab(group1[feature], group2[feature])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p

def t_test(group1, group2, feature):
    t_stat, p_value = stats.ttest_ind(group1[feature], group2[feature], equal_var=False)
    return t_stat, p_value

def z_test(group1, group2, feature):
    z_stat, p_value = ztest(group1[feature], group2[feature])
    return z_stat, p_value

def analyze_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return "Reject the null hypothesis (significant difference)"
    else:
        return "Fail to reject the null hypothesis (no significant difference)"

# Main function to perform A/B testing
def ab_testing(df):
    df['Risk'] = df['TotalClaims'] / df['TotalPremium']
    df['Margin'] = (df['TotalPremium'] - df['TotalClaims']) / df['TotalPremium']

    group_a_provinces = df[df['Province'].isin(['ProvinceA', 'ProvinceB'])]
    group_b_provinces = df[df['Province'].isin(['ProvinceC', 'ProvinceD'])]

    group_a_zip = df[df['PostalCode'].isin(['12345', '67890'])]
    group_b_zip = df[df['PostalCode'].isin(['54321', '09876'])]

    group_a_margin = df[df['PostalCode'].isin(['12345', '67890'])]
    group_b_margin = df[df['PostalCode'].isin(['54321', '09876'])]

    group_a_gender = df[df['Gender'] == 'Male']
    group_b_gender = df[df['Gender'] == 'Female']

    chi2_provinces, p_value_provinces = chi_squared_test(group_a_provinces, group_b_provinces, 'Risk')
    chi2_zip, p_value_zip = chi_squared_test(group_a_zip, group_b_zip, 'Risk')
    t_stat_margin, p_value_margin = t_test(group_a_margin, group_b_margin, 'Margin')
    chi2_gender, p_value_gender = chi_squared_test(group_a_gender, group_b_gender, 'Risk')

    results = {
        'Risk across Provinces': analyze_p_value(p_value_provinces),
        'Risk across Zip Codes': analyze_p_value(p_value_zip),
        'Margin differences across Zip Codes': analyze_p_value(p_value_margin),
        'Risk difference between Men and Women': analyze_p_value(p_value_gender),
    }

    return results
