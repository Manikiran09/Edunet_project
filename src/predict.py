import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_risk(features, feature_names, dropped_features, label_encoders):
    """Predict credit risk for given features."""
    try:
        # Load model and preprocessor
        logger.info("Loading model and preprocessor...")
        with open('C:/Users/sarve/Desktop/New folder/models/risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('C:/Users/sarve/Desktop/New folder/models/preprocessor.pkl', 'rb') as f:
            loaded_label_encoders, scaler, _ = pickle.load(f)
        
        # Convert features to DataFrame with original feature names
        original_features = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                             'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
                             'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']
        df = pd.DataFrame([features], columns=original_features)
        
        # Drop correlated features
        df = df.drop(dropped_features, axis=1, errors='ignore')
        
        # Cap outliers
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
        
        # Encode categorical features
        for col in df.columns:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except ValueError:
                    logger.warning(f"Unknown value in {col}. Using default encoding.")
                    df[col] = label_encoders[col].classes_[0]
        
        # Scale features
        X = scaler.transform(df)
        
        # Predict
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0][prediction]
        risk_label = 'Not Risky' if prediction == 0 else 'Risky'
        
        return f"Predicted Risk: {risk_label} (Confidence: {confidence:.2%})"
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return f"Prediction failed: {str(e)}"

def run_predictor():
    """Interactive prediction loop."""
    try:
        # Load feature names, dropped features, and label encoders
        with open('C:/Users/sarve/Desktop/New folder/data/processed/processed_data.pkl', 'rb') as f:
            _, _, _, _, feature_names = pickle.load(f)
        with open('C:/Users/sarve/Desktop/New folder/models/preprocessor.pkl', 'rb') as f:
            label_encoders, _, dropped_features = pickle.load(f)
        
        # Define valid categorical values
        categorical_info = {
            'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
            'loan_intent': ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'],
            'loan_grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'cb_person_default_on_file': ['Y', 'N']
        }
        
        print("Credit Risk Assessor. Type 'exit' to quit.")
        print("Enter 11 features, comma-separated, in order:")
        print("person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, "
              "loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length")
        print("Valid categorical values:")
        for col, values in categorical_info.items():
            print(f"{col}: {', '.join(values)}")
        print("Example: 20,9600,RENT,5.0,EDUCATION,B,1000,11.14,0.10,N,2")
        
        while True:
            user_input = input("Features: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            try:
                features = user_input.split(',')
                if len(features) != 11:
                    print(f"Please provide exactly 11 features, got {len(features)}.")
                    continue
                
                # Validate numerical features
                numerical_indices = [0, 1, 3, 6, 7, 8, 10]
                for idx in numerical_indices:
                    try:
                        float(features[idx])
                    except ValueError:
                        print(f"Invalid numerical value at position {idx+1}: '{features[idx]}'. Must be a number.")
                        raise ValueError("Invalid numerical input")
                
                # Validate categorical features
                categorical_indices = {2: 'person_home_ownership', 4: 'loan_intent', 5: 'loan_grade', 9: 'cb_person_default_on_file'}
                for idx, col in categorical_indices.items():
                    if features[idx].strip() not in categorical_info[col]:
                        print(f"Invalid value for {col}: '{features[idx]}'. Valid values: {', '.join(categorical_info[col])}")
                        raise ValueError("Invalid categorical input")
                
                result = predict_risk(features, feature_names, dropped_features, label_encoders)
                print(f"Result: {result}")
            except ValueError as ve:
                print(f"Error: {str(ve)}. Please try again.")
            except Exception as e:
                logger.error(f"Invalid input: {str(e)}")
                print(f"Unexpected error: {str(e)}. Please try again.")
    except Exception as e:
        logger.error(f"Error loading feature names: {str(e)}")
        print("Failed to initialize predictor.")

if __name__ == "__main__":
    run_predictor()