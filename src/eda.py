import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create outputs directory
os.makedirs('outputs/eda', exist_ok=True)

def perform_eda():
    """Main EDA function."""
    try:
        # Load dataset
        dataset_path = 'C:/Users/sarve/Desktop/New folder/data/raw/credit_risk_dataset.csv'
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at {dataset_path}. Please place credit_risk_dataset.csv in data/raw/")
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        logger.info("Loading dataset for EDA...")
        df = pd.read_csv(dataset_path)
        
        # Basic statistics
        logger.info("Generating summary statistics...")
        summary_stats = df.describe(include='all')
        summary_stats.to_csv('outputs/eda/summary_stats.csv')
        logger.info("Saved summary stats to outputs/eda/summary_stats.csv")
        
        # Class distribution
        logger.info("Analyzing class distribution...")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='loan_status', data=df)
        plt.title('Class Distribution (0: Not Risky, 1: Risky)')
        plt.xlabel('Loan Status')
        plt.ylabel('Count')
        plt.savefig('outputs/eda/class_distribution.png')
        plt.close()
        class_counts = df['loan_status'].value_counts()
        logger.info(f"Class distribution:\n{class_counts}")
        
        # Missing values
        logger.info("Checking for missing values...")
        missing_values = df.isnull().sum()
        with open('outputs/eda/missing_values.txt', 'w') as f:
            f.write(str(missing_values))
        logger.info("Saved missing values report to outputs/eda/missing_values.txt")
        
        # Numerical feature distributions
        logger.info("Plotting numerical feature distributions...")
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('loan_status')
        for col in numerical_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.savefig(f'outputs/eda/hist_{col}.png')
            plt.close()
        
        # Outlier detection (boxplots)
        logger.info("Detecting outliers...")
        for col in numerical_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x='loan_status', y=col, data=df)
            plt.title(f'Boxplot of {col} by Loan Status')
            plt.savefig(f'outputs/eda/boxplot_{col}.png')
            plt.close()
        
        # Correlation matrix
        logger.info("Computing correlation matrix...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig('outputs/eda/correlation_matrix.png')
        plt.close()
        
        # Categorical feature analysis
        logger.info("Analyzing categorical features...")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=col, hue='loan_status', data=df)
            plt.title(f'{col} vs Loan Status')
            plt.xticks(rotation=45)
            plt.savefig(f'outputs/eda/countplot_{col}.png')
            plt.close()
        
        print("EDA complete. Check outputs/eda/ for plots and reports.")
    except Exception as e:
        logger.error(f"Error in EDA: {str(e)}")
        raise

if __name__ == "__main__":
    perform_eda()