import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import logging
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

def evaluate_model():
    """Enhanced model evaluation for Logistic Regression and Random Forest."""
    try:
        # Load processed data
        logger.info("Loading processed data...")
        with open('C:/Users/sarve/Desktop/New folder/data/processed/processed_data.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test, feature_names = pickle.load(f)
        logger.info(f"Loaded data with {X_test.shape[1]} features: {list(feature_names)}")

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }
        
        # Metrics storage
        metrics_summary = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]  # Probability for positive class (Risky)
            
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
            cm = confusion_matrix(y_pred, y_test)
            tn, fp, fn, tp = cm.ravel()
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_prob)
            ap_score = average_precision_score(y_test, y_prob)
            
            # Store metrics
            metrics_summary[name] = {
                'Accuracy': accuracy,
                'Precision (Not Risky)': precision[0],
                'Recall (Not Risky)': recall[0],
                'F1-score (Not Risky)': f1[0],
                'Precision (Risky)': precision[1],
                'Recall (Risky)': recall[1],
                'F1-score (Risky)': f1[1],
                'ROC AUC': roc_auc,
                'Average Precision': ap_score,
                'Confusion Matrix': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
            }
            
            # Print metrics
            print(f"\nMetrics for {name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Not Risky (0): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}")
            print(f"Risky (1): Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Average Precision: {ap_score:.4f}")
            print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            # Confusion Matrix Plot
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Risky', 'Risky'], yticklabels=['Not Risky', 'Risky'])
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f'outputs/confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()
            logger.info(f"Saved confusion matrix to outputs/confusion_matrix_{name.lower().replace(' ', '_')}.png")
            
            # ROC Curve
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc="lower right")
            plt.savefig(f'outputs/roc_curve_{name.lower().replace(" ", "_")}.png')
            plt.close()
            logger.info(f"Saved ROC curve to outputs/roc_curve_{name.lower().replace(' ', '_')}.png")
            
            # Precision-Recall Curve
            plt.figure(figsize=(6, 4))
            plt.plot(recall_pr, precision_pr, color='purple', lw=2, label=f'Precision-Recall curve (AP = {ap_score:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {name}')
            plt.legend(loc="lower left")
            plt.savefig(f'outputs/precision_recall_curve_{name.lower().replace(" ", "_")}.png')
            plt.close()
            logger.info(f"Saved Precision-Recall curve to outputs/precision_recall_curve_{name.lower().replace(' ', '_')}.png")
            
            # Calibration Plot
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            plt.figure(figsize=(6, 4))
            plt.plot(prob_pred, prob_true, marker='o', color='green', label='Calibration curve')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Plot - {name}')
            plt.legend(loc="best")
            plt.savefig(f'outputs/calibration_plot_{name.lower().replace(" ", "_")}.png')
            plt.close()
            logger.info(f"Saved calibration plot to outputs/calibration_plot_{name.lower().replace(' ', '_')}.png")
            
            # Feature Importance (for Random Forest)
            if hasattr(model, 'feature_importances_'):
                logger.info(f"Plotting feature importance for {name}...")
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(importances)), importances[indices], align='center', color='skyblue')
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                plt.title(f'Feature Importance - {name}')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()
                plt.savefig(f'outputs/feature_importance_{name.lower().replace(" ", "_")}.png')
                plt.close()
                logger.info(f"Saved feature importance to outputs/feature_importance_{name.lower().replace(' ', '_')}.png")
        
        # Save metrics to file
        logger.info("Saving metrics summary...")
        with open('outputs/evaluation_metrics.txt', 'w') as f:
            for name, metrics in metrics_summary.items():
                f.write(f"\nMetrics for {name}:\n")
                f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
                f.write(f"Not Risky (0): Precision={metrics['Precision (Not Risky)']:.4f}, "
                        f"Recall={metrics['Recall (Not Risky)']:.4f}, F1={metrics['F1-score (Not Risky)']:.4f}\n")
                f.write(f"Risky (1): Precision={metrics['Precision (Risky)']:.4f}, "
                        f"Recall={metrics['Recall (Risky)']:.4f}, F1={metrics['F1-score (Risky)']:.4f}\n")
                f.write(f"ROC AUC: {metrics['ROC AUC']:.4f}\n")
                f.write(f"Average Precision: {metrics['Average Precision']:.4f}\n")
                f.write(f"Confusion Matrix: TN={metrics['Confusion Matrix']['TN']}, "
                        f"FP={metrics['Confusion Matrix']['FP']}, "
                        f"FN={metrics['Confusion Matrix']['FN']}, "
                        f"TP={metrics['Confusion Matrix']['TP']}\n")
        logger.info("Saved metrics to outputs/evaluation_metrics.txt")
        
        print("\nEvaluation complete. Metrics saved to outputs/evaluation_metrics.txt")
        print("Plots saved to outputs/ (confusion matrix, ROC, Precision-Recall, Calibration, feature importance)")
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()