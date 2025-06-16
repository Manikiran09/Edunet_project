# 🏦 Loan Risk Classifier

A machine learning system to classify loan applications as **risky** or **not risky** based on historical data. This project includes data preprocessing, exploratory data analysis (EDA), training, prediction, and evaluation.

---

## 📂 Project Structure
LoanRiskClassifier/
├── data/ # Raw and processed datasets
├── src/ # Python modules
│ ├── preprocess.py # Data cleaning & preprocessing
│ ├── eda.py # Visualizations and data profiling
│ ├── train.py # ML model training
│ ├── predict.py # Prediction script
│ └── evaluate.py # Model evaluation metrics
├── models/ # Saved ML models
├── outputs/ # Logs, reports, predictions
├── run.py # Central command runner
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Setup


# 1. Clone the repository
git clone https://github.com/Manikiran09/Edunet_project

cd Edunet_project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

#4. To deploy
i)To preprocess Raw Data
python run.py preprocess
ii)For Exploratory and Data Analysis
python run.py eda
iii)For training model
python run.py train
iv)for prediction
python run.py predict
v)For Evalution of model
python run.py evaluate

