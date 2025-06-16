import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    """Main training function with balanced class weights."""
    try:
        # Load processed data
        logger.info("Loading processed data...")
        with open('C:/Users/sarve/Desktop/New folder/data/processed/processed_data.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test, _ = pickle.load(f)
        logger.info("Loaded processed data")
        
        # Validate class distribution
        train_classes = np.unique(y_train)
        logger.info(f"Training classes: {train_classes}")
        if len(train_classes) < 2:
            logger.error("Training data contains only one class. At least two classes (0 and 1) are required.")
            raise ValueError("Training data contains only one class.")
        
        logger.info(f"Train label distribution:\n{pd.Series(y_train).value_counts()}")
        
        # Train models with balanced class weights
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }
        
        best_model = None
        best_f1 = 0
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"\nResults for {name}:")
            print(classification_report(y_test, y_pred, target_names=['Not Risky', 'Risky']))
            
            report = classification_report(y_test, y_pred, output_dict=True)
            f1_risky = report['1']['f1-score']  # 1: Risky
            if f1_risky > best_f1:
                best_f1 = f1_risky
                best_model = model
            logger.info(f"Completed training {name}")
        
        if best_model is None:
            logger.error("No models trained successfully.")
            raise RuntimeError("No models trained successfully.")
        
        # Save best model
        logger.info("Saving best model...")
        with open('C:/Users/sarve/Desktop/New folder/models/risk_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Best model saved: {best_model.__class__.__name__}")
        
        print("Training complete. Check logs for details.")
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()