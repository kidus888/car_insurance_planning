
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

def txt_to_csv(txt_file_path, csv_file_path, delimiter='|'):
   
    try:
        # Read the .txt file into a DataFrame, adjusting the delimiter as needed
        df = pd.read_csv(txt_file_path, delimiter=delimiter)
        
        # Check if the file was loaded properly as a DataFrame
        if isinstance(df, pd.DataFrame):
            # Write the DataFrame to a .csv file
            df.to_csv(csv_file_path, index=False)
            print(f"File successfully converted and saved as {csv_file_path}")
        else:
            print("The file could not be loaded as a DataFrame. Please check the delimiter or file format.")
        
    except Exception as e:
        print(f"Error occurred: {e}")


def load_data(file_path):
    """Load data from a CSV or other file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def data_summary(df):
    """Perform data summarization by calculating descriptive statistics."""
    summary = df.describe(include='all')
    return summary

def check_missing_values(df):
    """Check for missing values in the dataframe."""
    missing = df.isnull().sum()
    return missing

def plot_histograms(df, columns):
    """Plot histograms for numerical columns."""
    df[columns].hist(bins=50, figsize=(20, 15))
    plt.show()

def plot_categorical_bar(df, columns):
    """Plot bar charts for categorical columns."""
    for col in columns:
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution of {col}")
        plt.show()


def calculate_correlation(df, columns):
    """Calculate and plot correlation matrix for the specified columns."""
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    return corr_matrix

def scatter_plot(df, x_column, y_column):
    """Plot scatter plot between two columns."""
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.title(f"Scatter plot: {x_column} vs {y_column}")
    plt.show()


def box_plot(df, column):
    """Create box plots for numerical data to detect outliers."""
    sns.boxplot(data=df, x=column)
    plt.title(f"Boxplot for {column}")
    plt.show()
