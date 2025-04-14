# Methodology

## Introduction

This project investigates youth substance use patterns using data from the **National Survey on Drug Use and Health (NSDUH)**. We address three predictive tasks:

1. **Binary Classification**: Predict cigarette use initiation
2. **Multi-Class Classification**: Categorize marijuana use frequency
3. **Regression**: Estimate age of first cigarette use

---

## Data Cleaning

### Raw Data Overview
The NSDUH dataset contains 32,900+ responses with 200+ variables spanning substance use behaviors, demographics, and social factors. Key challenges included:
- **Missing Values**: 12% missingness in critical columns (PARHLPHW, SCHFELT)
- **Special Codes**: 991 (Never Used), 993 (Did Not Use Recently) in substance use variables
- **Class Imbalance**: Only 14.2% of youth reported cigarette use

### Cleaning Process
1. **Special Code Handling**:
Example code for handling special values
```
df['IRCIGAGE'] = df['IRCIGAGE'].replace(991, np.nan)
df = df[df['IRCIGAGE'] < 900] # Remove non-users for regression
```



2. **Imputation**:
- Used mean imputation for continuous variables (INCOME, SCHFELT)
- Retained 'Never Used' codes (91/93) as separate categories

3. **Train-Test Split**:
- Stratified sampling preserved class distributions
- 80-20 split with random seed for reproducibility

---

## Feature Engineering

### Strategy
We employed domain-inspired feature engineering to enhance predictive power:

1. **Interaction Terms**:
- `SCHOOL_SATISFACTION = SCHFELT * AVGGRADE`
- Captures combined school environment effects
- **Rationale**: Research shows school climate moderates substance use risk

2. **Composite Variables**:
```
df['RISK_TAKING'] = YOFIGHT2 + YOGRPFT2 # Physical fights + group fights
```
- Combines correlated risk behaviors into stable predictors

3. **Standardized Measures**:
- `STNDALC` (alcohol attitudes) preserved as ordinal scale
- Enables comparison across substance types

### Validation
- Variance Inflation Factor (VIF) < 5 for all features
- Feature correlation < 0.7 (avoid multicollinearity)

---

## Modeling Approach

### Decision Tree Selection Rationale
Chosen for:
1. Interpretability (critical for public health applications)
2. Native handling of mixed data types
3. Resistance to outliers in survey data 

### Binary Classification (Cigarette Use)
**Architecture**:
```
DecisionTreeClassifier(
max_depth=4, # Limits overfitting while capturing key splits
min_samples_leaf=5, # Ensures stable probability estimates
class_weight='balanced'
)
```

**Key Design Choices**:
- SMOTE oversampling addresses 6:1 class imbalance
- Depth limit set via cross-validation plateau analysis
- Balanced class weights improve minority class recall

### Multi-Class Classification (Marijuana Frequency)
**Tree Structure**:
- 63 total nodes (32 leaves)
- Max depth = 5
- **Rationale**: Deeper than binary model to capture frequency nuances

**Splitting Criteria**:
- Gini impurity minimization
- Cost-complexity pruning (α=0.01) prevents overfitting

### Regression (Age of First Use)
**Model Configuration**:
```
DecisionTreeRegressor(
max_depth=5,
min_samples_leaf=10, # Larger leaves for continuous target stability
ccp_alpha=0.02 # From grid search optimization
)
```

**Performance**:
- MSE: 5.8579 → Average error of ±2.4 years
- R²: 0.0328 highlights prediction difficulty with available features

---

## Model Interpretation

### Tree Structural Analysis
| Model                | Nodes | Leaves | Depth | Avg. Samples/Leaf |
|----------------------|-------|--------|-------|-------------------|
| Binary Classification| 25    | 13     | 4     | 412              |
| Multi-Class          | 63    | 32     | 5     | 158              |
| Regression           | 53    | 27     | 5     | 25               |

**Depth Selection**:
- Limited to 5 via early stopping to maintain clinical interpretability
- Deeper trees (>7 levels) showed validation performance degradation

### Feature Importance
Used SHAP values for unified interpretation across models:
1. **Parental Support** (0.38 SHAP) - Most impactful across all models
2. **Income** (0.21) - Particularly salient for early initiation
3. **Peer Influence** (0.19) - Mediates school environment effects

---

## Ethical Considerations

1. **Bias Mitigation**:
   - Fairness constraints in tree splitting 
   - Demographic parity checks across gender/race subgroups

2. **Privacy Preservation**:
   - Aggregated results at county level
   - Suppressed small cell counts (<10) per NSDUH guidelines 

3. **Actionable Insights**:
   - Focused on modifiable predictors (school climate, parental monitoring)
   - Avoided racial/economic profiling in interpretation