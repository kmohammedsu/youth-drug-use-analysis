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
- **Special Codes Handling**: Replaced special codes (e.g., `97`, `98`,`994`,`997`,`998`,`999`: Missing, refused, blank, skip set to NaN). Retained codes `91` ("Never Used") and `93` ("Did Not Use Recently").
- **Target Variable Creation**:
  - Binary target variable for cigarette use prediction (`cigarette_use`).
  - Multi-class target variable for marijuana frequency prediction (`marijuana_freq`).
  - Continuous target variable for age of first cigarette use prediction (`IRCIGAGE`).
- **Missing Value Handling**: Dropped rows with missing values in key columns like `IRCIGAGE`, `PARHLPHW`, `SCHFELT`, and `INCOME`.

### 2. Feature Engineering
To capture meaningful relationships between variables:
- **Interaction Terms**:
  - `School_Parental_Interaction = dParental_Support * School_Safety`
  - `Income_Risk_Interaction = Income_Level * Risk_Behavior`
- **Composite Variables**:
    - `School_Parental_Interaction = Parental_Support × School_Safety`
    - `Peer_Influence = Peer_Use_Ever × Peer_Use_Month`
- **Temporal Features** (for Regression):
  - `Years_Since_First_Drink = Cigarette_Age_of_First_Use - Alcohol_age_of_First_Use`
  - `School_Satisfaction = Going_to_school_Past_Year × AVG_GRADE`
  - `Parental_Support = Parents_with_Homework × Parents_Check_If_Homewoke_Done`
  - `Peer_Influence = CloseFriend_feel_Abt_Smoking × CloseFriend_feel_Abt_Marijuana`
  - `Risk_Taking = Youth_had_Serious_Fight + Youth_had_Serious_Fight_With_Group`
  - `Income_Access = INCOME × PRPKCIG2`
  - `Peer_Parent_Interaction = Peer_Influence × Parental_Support`
  - `School_Risk_Interaction = Going_to_school_Past_Year × Risk_Taking`
  - `Total_Risk = Youth_had_Serious_Fight + Youth_had_Serious_Fight_With_Group + Youth_Indulging in_Stealing`
  - `Years_Since_First_Drink = Cigarette_Age_of_First_Use  - Alcohol_age_of_First_Use`
  - `Income_Squared = INCOME²`
  - `Peer_Influence_Squared = Peer_Influence²`

---

## 🔍 Modeling Approach

We compared three machine learning algorithms across all prediction tasks:

### Binary Classification (Cigarette Use)
- **Objective**: Predict whether a youth has ever smoked cigarettes.
- **Algorithms**: Decision Tree, Bagging, Random Forest, Gradient Boosting
- **Tree Structure** (Decision Tree):
  - Number of Nodes: 25
  - Number of Leaf Nodes: 13
  - Maximum Depth: 4
- **Key Metrics**:

| Model             | Accuracy | Precision | Recall | F1   | ROC AUC |
|-------------------|----------|-----------|--------|------|---------|
| Decision Tree     | 0.694    | 0.122     | 0.520  | 0.198| 0.645   |
| Bagging           | 0.627    | 0.114     | 0.612  | 0.192| 0.642   |
| Random Forest     | 0.634    | 0.115     | 0.605  | 0.193| 0.641   |
| Gradient Boosting | 0.642    | 0.117     | 0.599  | 0.195| 0.644   |

- **Key Insights**:
  - Interaction terms dominate feature importance, with **School_Parental_Interaction** and **Income_Risk_Interaction** together accounting for over 60% of model importance.
  - Decision Tree achieves the highest accuracy and ROC AUC, while Bagging provides the best recall.

### Multi-Class Classification (Marijuana Frequency)
- **Objective**: Predict marijuana use frequency categories (Never, Seldom, Sometimes, Frequent).
- **Algorithms**: Decision Tree, Bagging, Random Forest, Gradient Boosting
- **Tree Structure** (Decision Tree):
  - Number of Nodes: 63
  - Number of Leaf Nodes: 32
  - Maximum Depth: 5
- **Key Metrics**:

| Model             | Accuracy | Macro F1 | Weighted F1 |
|-------------------|----------|----------|-------------|
| Decision Tree     | 0.729    | 0.350    | 0.770       |
| Bagging           | 0.729    | 0.349    | 0.770       |
| Random Forest     | 0.718    | 0.337    | 0.764       |
| Gradient Boosting | 0.744    | 0.359    | 0.777       |

- **Key Insights**:
  - **Peer_Influence** is the dominant predictor (over 50% importance).
  - Gradient Boosting provides the best overall performance across all metrics.

### Regression (Age of First Cigarette Use)
- **Objective**: Predict the age at which youth first smoke cigarettes.
- **Algorithms**: Decision Tree, Bagging, Random Forest, Gradient Boosting
- **Tree Structure** (Decision Tree):
  - Number of Nodes: 33
  - Number of Leaf Nodes: 17
  - Maximum Depth: 5
- **Key Metrics**:

| Model             | RMSE     | R²       |
|-------------------|----------|----------|
| Decision Tree     | 1.986    | 0.438    |
| Bagging           | 1.944    | 0.461    |
| Random Forest     | 1.940    | 0.464    |
| Gradient Boosting | 2.038    | 0.408    |

- **Key Insights**:
  - **Years_Since_First_Drink** is the most important feature (over 70% importance).
  - Random Forest provides the best predictive performance with the lowest RMSE and highest R².

---

## 📂 Project Structure
```
youth-drug-use-analysis/
├── data/
│ ├── youth_data.csv
│ ├── processed_data.csv
├── notebooks/
│ ├── binary_classification.ipynb
│ ├── multi_class_classification.ipynb
│ ├── regression.ipynb
├── results/
│ ├── Decision_Tree_confusion_matrix.png
│ ├── Bagging_confusion_matrix.png
│ ├── Random_Forest_confusion_matrix.png
│ ├── Gradient_Boosting_confusion_matrix.png
│ ├── Decision_Tree_actual_vs_predicted.png
│ ├── Random_Forest_feature_importance.png
│ ├── best_model_feature_importance.png
├── README.md
└── requirements.txt
```
---

## 🚀 How to Reproduce

1. **Clone the Repository**
    ```
    git clone https://github.com/yourusername/youth-drug-use-analysis.git
    cd youth-drug-use-analysis
    ```
2. **Set Up a Virtual Environment**
    ```
    python -m venv venv
    ```
3. **Activate the Virtual Environment**
    - On Windows:
      ```
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```
      source venv/bin/activate
      ```
4. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```
5. **Run the Notebooks**
    ```
    jupyter notebook notebooks/
    ```

---

## 🔮 Future Work

1. **Advanced Models**: Explore neural networks or stacked ensemble methods.
2. **Feature Extraction**: Develop techniques for richer categorical feature extraction.
3. **Longitudinal Analysis**: Use multi-wave data to track substance use patterns over time.
4. **Explainable AI**: Apply SHAP or LIME for deeper model interpretation.
5. **Causal Inference**: Move beyond prediction to causal models for intervention design.

---

## 📚 References

- **NSDUH-2020-DS0001-info-codebook-1.pdf**: Official codebook for variable definitions and coding.
- **Project PDF**: "Predicting Youth Substance Use: A Decision Tree Approach" (2024).
- [SAMHSA. National Survey on Drug Use and Health (NSDUH)](https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/datafiles/2020)
- [Scikit-learn documentation: Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)


---

**Prepared:** April 2025  
**Author:** Khaja Moinuddin Mohammed   
**Course:** Data 5322 ML 2
---

📫 Reach out: 💼[LinkedIn](https://linkedin.com/in/emkaymoin) • 📧[Email](mailto:moinuddin0518@example.com)