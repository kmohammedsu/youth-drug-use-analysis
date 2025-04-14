import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
# Load cleaned data
df = pd.read_csv("data/processed_data.csv")

# Create target variable for marijuana frequency categories
df['marijuana_freq'] = pd.cut(
    df['IRMJFY'], 
    bins=[-1, 0, 30, 60, float('inf')], 
    labels=['Never', 'Seldom', 'Sometimes', 'Frequent']
)

# Feature engineering
df['SCHOOL_FAMILY'] = df['PARHLPHW'] * df['SCHFELT']  # Interaction between parental help and school safety
df['PEER_INFLUENCE'] = df['FRDMEVR2'] * df['FRDMJMON']  # Peer influence composite

# Handle missing values - required before SMOTE
imputer = SimpleImputer(strategy='mean')

# Define features and target variable for multi-class classification
X = df[['PARHLPHW', 'SCHFELT', 'INCOME', 'SCHOOL_FAMILY', 'PEER_INFLUENCE', 'YOGRPFT2', 'HEALTH2']]
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df['marijuana_freq']  # Target: marijuana use frequency categories

# Split data with stratification to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a decision tree classifier with parameters tuned for multi-class classification
clf = DecisionTreeClassifier(
    max_depth=5,  # Slightly deeper tree for multi-class problem
    min_samples_leaf=10,  # Require at least 10 samples per leaf for stability
    class_weight='balanced',  # Adjust for class imbalance
    random_state=42
)
clf.fit(X_train_resampled, y_train_resampled)

# Print tree information (number of nodes, leaves, and depth)
num_nodes = clf.tree_.node_count
num_leaves = clf.get_n_leaves()
max_depth = clf.tree_.max_depth

print("\nDecision Tree Information:")
print(f"Number of Nodes: {num_nodes}")
print(f"Number of Leaf Nodes: {num_leaves}")
print(f"Maximum Depth: {max_depth}")

# Make predictions on test set
y_pred = clf.predict(X_test)

# Evaluate model performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create and save confusion matrix
plt.figure(figsize=(10, 8))
cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix for Marijuana Use Frequency Prediction")
plt.savefig("results/multi_class_confusion_matrix.png")
plt.close()

# Visualize the decision tree
os.makedirs("results", exist_ok=True)  # Ensure results folder exists
plt.figure(figsize=(25,15))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Never', 'Seldom', 'Sometimes', 'Frequent'],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=12
)
plt.title("Decision Tree for Marijuana Use Frequency Prediction", fontsize=20)
plt.savefig("results/multi_class_decision_tree.png", bbox_inches='tight')
plt.close()

# Print feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance)

# Plot feature importance with better formatting
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('Feature Importance for Marijuana Use Frequency')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig("results/multi_class_feature_importance.png")
plt.close()

print("\nAll visualizations and metrics saved to the 'results' folder.")
