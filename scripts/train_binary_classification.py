import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/processed_data.csv")

# Handle missing values - required before SMOTE
imputer = SimpleImputer(strategy='mean')

# Define features and target variable for binary classification
X = df[['PARHLPHW', 'SCHFELT', 'INCOME']]  # Key features from decision tree
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df['cigarette_use']  # Binary target: 0=Never Used, 1=Ever Used

# Split data with stratification to preserve class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a decision tree classifier with parameters tuned for balance
clf = DecisionTreeClassifier(
    max_depth=4,  # Limit depth to prevent overfitting
    min_samples_leaf=5,  # Require at least 5 samples per leaf
    class_weight='balanced',  # Further adjust for class imbalance
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
cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("results/binary_confusion_matrix.png")
plt.close()

# Visualize the decision tree
os.makedirs("results", exist_ok=True)  # Ensure results folder exists
plt.figure(figsize=(25,15))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Never Used', 'Ever Used'],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=12
)
plt.title("Decision Tree for Cigarette Use Prediction", fontsize=20)
plt.savefig("results/binary_decision_tree.png", bbox_inches='tight')
plt.close()

# Print feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance)
