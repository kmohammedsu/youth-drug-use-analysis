# 🧠 Understanding Youth Substance Use: A Data-Driven Investigation

Substance use among youth is a complex issue influenced by family, peers, school environments, and personal attitudes. This project applies machine learning to uncover patterns in youth substance use behaviors using survey data from the **National Survey on Drug Use and Health (NSDUH)**.

Our analysis focuses on three key questions:
1. **Cigarette Use Prediction (Binary Classification)**: What factors predict whether a youth has ever smoked cigarettes?
2. **Marijuana Use Frequency Prediction (Multi-Class Classification)**: How do demographic and social factors influence marijuana use frequency?
3. **Age of First Cigarette Use Prediction (Regression)**: Can we predict the age at which youth first smoke cigarettes based on family, peer, and school-related factors?

---

## 📊 Dataset Overview

The dataset contains responses from youth about their substance use behaviors, demographic characteristics, and social influences. It includes:
- **Substance Use Variables**: Cigarette frequency (`IRCIGFM`), marijuana frequency (`IRMJFY`), age of first cigarette use (`IRCIGAGE`).
- **Demographic Variables**: Income level (`INCOME`), gender (`IRSEX`), race/ethnicity (`NEWRACE2`).
- **Social Variables**: Parental involvement (`PARHLPHW`), school safety (`SCHFELT`), peer influences (`FRDMEVR2`, `FRDMJMON`).

Key challenges included:
- Missing values in critical columns.
- Special codes indicating "unknown" or "refused" responses.
- Severe class imbalance in target variables.

---

## 🛠️ Methodology

### 1. Data Cleaning
To ensure high-quality data for analysis:
- **Special Codes Handling**: Replaced special codes (`97`, `98`) with `NaN`. Retained codes `91` ("Never Used") and `93` ("Did Not Use Recently").
- **Target Variable Creation**:
  - Binary target variable for cigarette use prediction (`cigarette_use`).
  - Multi-class target variable for marijuana frequency prediction (`marijuana_freq`).
  - Continuous target variable for age of first cigarette use prediction (`IRCIGAGE`).
- **Missing Value Handling**: Dropped rows with missing values in key columns like `IRCIGAGE`, `PARHLPHW`, `SCHFELT`, and `INCOME`.

### 2. Feature Engineering
To capture meaningful relationships between variables:
- **Interaction Terms**:
  - ```
    df['SCHOOL_SATISFACTION'] = df['SCHFELT'] * df['AVGGRADE']
    ```
    Captures combined effects of school safety and academic performance.
  - ```
    df['PARENTAL_SUPPORT'] = df['PARHLPHW'] * df['PARCHKHW']
    ```
    Represents the combined influence of parental help and monitoring.
- **Composite Variables**:
  - ```
    df['PEER_INFLUENCE'] = df['FRDPCIG2'] * df['FRDMEVR2']
    ```
    Combines peer cigarette use and marijuana use into a single predictor.
  - ```
    df['RISK_TAKING'] = df['YOFIGHT2'] + df['YOGRPFT2']
    ```
    Aggregates risk-taking behaviors such as fighting and group participation.

---

## 🔍 Modeling Approach

### Why Decision Trees?
Decision trees were chosen for their interpretability, ability to handle mixed data types, and resistance to outliers. They allow us to identify key predictors in a transparent way, which is critical for public health applications.

### Binary Classification (Cigarette Use)
- **Objective**: Predict whether a youth has ever smoked cigarettes.
- **Algorithm**: Decision Tree Classifier with SMOTE to address class imbalance.
- **Tree Structure**:
  - Number of Nodes: 25
  - Number of Leaf Nodes: 13
  - Maximum Depth: 4
- **Key Metrics**:
  - Accuracy: 65%
  - Precision for "Ever Used": 2%
  - Recall for "Ever Used": 52%
- **Key Insights**:
  - Parental involvement was the most influential predictor (54.8% importance).
  - School safety contributed significantly (31.1%).

### Multi-Class Classification (Marijuana Frequency)
- **Objective**: Predict marijuana use frequency categories (Never, Seldom, Sometimes, Frequent).
- **Algorithm**: Decision Tree Classifier with SMOTENC to handle categorical features and class imbalance.
- **Tree Structure**:
  - Number of Nodes: 63
  - Number of Leaf Nodes: 32
  - Maximum Depth: 5
- **Key Metrics**:
  - Accuracy: 75%
  - Precision for "Seldom": 17%
  - Recall for "Seldom": 53%
- **Key Insights**:
  - Peer influence was the dominant predictor (43.9% importance).
  - Health status and risk behaviors also played significant roles.

### Regression (Age of First Cigarette Use)
- **Objective**: Predict the age at which youth first smoke cigarettes.
- **Algorithm**: Decision Tree Regressor with polynomial features to capture non-linear relationships.
- **Tree Structure**:
  - Number of Nodes: 53
  - Number of Leaf Nodes: 27
  - Maximum Depth: 5
- **Key Metrics**:
  - Mean Squared Error (MSE): ~5.8579
  - Root Mean Squared Error (RMSE): ~2.4203
  - R² Score: ~0.0328
- **Key Insights**:
  - Income level was the most influential predictor.
  - Peer influence and standardized alcohol attitudes were also important.
  - Risk-taking behaviors contributed meaningfully to predictions.

---

## 📂 Project Structure
```
youth-drug-use-analysis/
├── data/
│ ├── youth_data.csv # Raw dataset
│ ├── processed_data.csv # Cleaned dataset used for modeling
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory Data Analysis notebook
│ ├── 02_data_cleaning.ipynb # Interactive documentation of data cleaning process
│ ├── 03_modeling.ipynb # Modeling summary notebook (binary, multi-class, regression)
├── scripts/
│ ├── data_cleaning.py # Script for cleaning raw data and generating processed dataset
│ ├── train_binary_classification.py # Binary classification model (cigarette use)
│ ├── train_multi_class_classification.py # Multi-class classification model (marijuana frequency)
│ ├── train_regression.py # Regression model (age of first cigarette use)
├── results/
│ ├── binary_decision_tree.png # Decision tree visualization for binary classification
│ ├── multi_class_decision_tree.png # Decision tree visualization for multi-class classification
│ ├── regression_decision_tree.png # Decision tree visualization for regression
│ ├── binary_confusion_matrix.png # Confusion matrix for binary classification
│ ├── multi_class_confusion_matrix.png # Confusion matrix for multi-class classification
│ ├── regression_predicted_vs_actual.png # Actual vs predicted scatter plot for regression
│ ├── regression_residual_plot.png # Residual plot for regression errors
├── presentation/
│ ├── slides.pptx # Slide deck summarizing findings and methodology
│ └── presentation_script.txt # Optional narration script for video presentation
├── README.md # Project overview, methodology, and instructions
└── requirements.txt # Python dependencies list

```

---

## 🚀 How to Reproduce

Follow these steps to set up and reproduce the analysis:

Clone the Repository
```
git clone https://github.com/yourusername/youth-drug-use-analysis.git

cd youth-drug-use-analysis
```
Set Up a Virtual Environment
```
python -m venv venv
```
Activate Virtual Environment on Windows:
```
venv\Scripts\activate
```
Or on macOS/Linux:
```
source venv/bin/activate
```
Install Dependencies
```
pip install -r requirements.txt
```
Run Data Cleaning Scrip
```
python scripts/data_cleaning.py
```
Run Models Sequentially:
```
python scripts/train_binary_classification.py # Binary Classification Model
python scripts/train_multi_class_classification.py # Multi-Class Model
python scripts/train_regression.py # Regression Model

```

---



## 🔮 Future Work

1. Expand analysis to other substances (e.g., opioids or vaping behaviors)`.
2. Incorporate additional features such as social media behavior patterns.
3. Explore ensemble models like Random Forests or Gradient Boosting.

---