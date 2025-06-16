import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        logger.error("Please specify a command: eda, preprocess, train, evaluate, predict")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == 'eda':
            logger.info("Starting EDA...")
            from src.eda import perform_eda
            perform_eda()
        elif command == 'preprocess':
            logger.info("Starting preprocessing...")
            from src.preprocess_data import preprocess_data
            preprocess_data()
        elif command == 'train':
            logger.info("Starting training...")
            from src.train_model import train_model
            train_model()
        elif command == 'evaluate':
            logger.info("Starting evaluation...")
            from src.evaluate_model import evaluate_model
            evaluate_model()
        elif command == 'predict':
            logger.info("Starting prediction...")
            from src.predict import run_predictor
            run_predictor()
        else:
            logger.error(f"Unknown command: {command}. Use eda, preprocess, train, evaluate, or predict")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing {command}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()