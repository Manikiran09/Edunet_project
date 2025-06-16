import pandas as pd
import os
import logging
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_data():
    """Main preprocessing function with EDA insights."""
    try:
        # Load dataset
        dataset_path = 'C:/Users/sarve/Desktop/New folder/data/raw/credit_risk_dataset.csv'
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at {dataset_path}. Please place credit_risk_dataset.csv in data/raw/")
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        logger.info("Loading dataset...")
        df = pd.read_csv(dataset_path)
        
        # Handle missing values (impute with mode/median as per EDA)
        logger.info("Handling missing values...")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Cap outliers (based on EDA boxplots, e.g., 99th percentile for income)
        logger.info("Capping outliers...")
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('loan_status')
        for col in numerical_cols:
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
        
        # Encode categorical features
        logger.info("Encoding categorical features...")
        label_encoders = {}
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Validate class distribution
        label_counts = df['loan_status'].value_counts()
        logger.info(f"Label distribution:\n{label_counts}")
        if len(label_counts) < 2:
            logger.error("Dataset contains only one class. At least two classes (0 and 1) are required.")
            raise ValueError("Dataset contains only one class.")
        
        # Feature selection (drop highly correlated features, e.g., loan_percent_income if correlated)
        logger.info("Performing feature selection...")
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        df = df.drop(to_drop, axis=1)
        logger.info(f"Dropped correlated features: {to_drop}")
        
        # Split features and target
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
        feature_names = X.columns
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train label distribution:\n{pd.Series(y_train).value_counts()}")
        logger.info(f"Test label distribution:\n{pd.Series(y_test).value_counts()}")
        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Save processed data and preprocessing objects
        logger.info("Saving processed data...")
        with open('C:/Users/sarve/Desktop/New folder/data/processed/processed_data.pkl', 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test, feature_names), f)
        with open('C:/Users/sarve/Desktop/New folder/models/preprocessor.pkl', 'wb') as f:
            pickle.dump((label_encoders, scaler, to_drop), f)
        logger.info("Saved processed data and preprocessor")
        
        print("Preprocessing complete. Check logs for details.")
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()