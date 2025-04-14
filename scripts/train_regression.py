import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load cleaned data

df = pd.read_csv("data/processed_data.csv")

# Filter to only include respondents who have ever used cigarettes (IRCIGAGE != 991)
print("Filtering data for cigarette users...")
df_users = df[df['IRCIGAGE'] < 900]  # Filter out non-users (code 991)

print(f"Number of cigarette users in dataset: {len(df_users)}")
print(f"Age of first use range: {df_users['IRCIGAGE'].min()} to {df_users['IRCIGAGE'].max()} years")

# Feature engineering
df_users['SCHOOL_SATISFACTION'] = df_users['SCHFELT'] * df_users['AVGGRADE']  # School satisfaction composite
df_users['PARENTAL_SUPPORT'] = df_users['PARHLPHW'] * df_users['PARCHKHW']  # Parental support composite
df_users['PEER_INFLUENCE'] = df_users['FRDPCIG2'] * df_users['FRDMEVR2']  # Peer cigarette influence
df_users['RISK_TAKING'] = df_users['YOFIGHT2'] + df_users['YOGRPFT2']  # Risk-taking behavior composite

# Select features and target variable
features = [
    'SCHOOL_SATISFACTION', 'PARENTAL_SUPPORT', 'PEER_INFLUENCE', 'RISK_TAKING',
    'STNDSCIG', 'STNDSMJ', 'STNDALC', 'HEALTH2', 'YTHACT2', 'PRGDJOB2', 
    'PRLMTTV2', 'RLGIMPT', 'INCOME'
]

# Drop rows with missing values in features or target
X = df_users[features].dropna()
y = df_users.loc[X.index, 'IRCIGAGE']  # Ensure alignment with features

# Print information about the data
print(f"Number of samples after dropping missing values: {len(X)}")
print(f"Target variable distribution: Mean = {y.mean():.2f}, Median = {y.median():.2f}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Train Decision Tree Regressor
print("Training Decision Tree Regressor...")
dt_regressor = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
dt_regressor.fit(X_train, y_train)

# Print tree information (number of nodes, leaves, and depth)
num_nodes = dt_regressor.tree_.node_count
num_leaves = dt_regressor.get_n_leaves()
max_depth = dt_regressor.tree_.max_depth

print("\nDecision Tree Information:")
print(f"Number of Nodes: {num_nodes}")
print(f"Number of Leaf Nodes: {num_leaves}")
print(f"Maximum Depth: {max_depth}")

# Evaluate model performance
y_pred = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Create visualizations

# 1. Decision Tree Visualization
plt.figure(figsize=(25,15))
plot_tree(
    dt_regressor,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=12
)
plt.title("Decision Tree for Age of First Cigarette Use Prediction", fontsize=20)
plt.savefig("results/regression_decision_tree.png", bbox_inches='tight')
plt.close()

# 2. Feature Importance Plot
plt.figure(figsize=(12, 8))
importance = pd.Series(dt_regressor.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).plot(
    kind='bar', 
    color='skyblue', 
    edgecolor='black'
)
plt.title("Feature Importance for Age of First Cigarette Use Prediction", fontsize=16)
plt.ylabel("Importance Score", fontsize=14)
plt.xlabel("Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/regression_feature_importance.png")
plt.close()

# 3. Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, edgecolor='k')
sns.regplot(x=y_test, y=y_pred, scatter=False, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel("Actual Age of First Cigarette Use", fontsize=14)
plt.ylabel("Predicted Age of First Cigarette Use", fontsize=14)
plt.title("Actual vs Predicted Age of First Cigarette Use", fontsize=16)
plt.grid(True, alpha=0.3)
plt.savefig("results/regression_predicted_vs_actual.png")
plt.close()

# 4. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Age of First Cigarette Use", fontsize=14)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=14)
plt.title("Residual Plot for Age of First Cigarette Use Prediction", fontsize=16)
plt.grid(True, alpha=0.3)
plt.savefig("results/regression_residual_plot.png")
plt.close()

print("\nAll visualizations saved to the 'results' folder")
