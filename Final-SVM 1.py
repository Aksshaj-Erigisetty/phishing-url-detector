import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Load the data
phishing_data = pd.read_csv('cleaned-outliers.csv')

# Start timing
start_time = time.time()

# Features and target
features = ['NoOfImage', 'NoOfSelfRef', 'LineOfCode', 'NoOfExternalRef']
target = 'label'

# Split the data into features (X) and target (y)
X = phishing_data[features].values
y = phishing_data[target].values

# Split into training and test sets (final evaluation set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler and model
scaler = StandardScaler()
svc = SVC(probability=True)

# Set up 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# To store metrics for cross-validation
mean_fpr = np.linspace(0, 1, 100)  # Common FPR for averaging
tprs = []  # True Positive Rates for each fold
roc_aucs = []  # AUC scores for each fold
conf_matrices = []  # Confusion matrices for each fold
classification_reports = []  # Classification reports for each fold

# Cross-validation process
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    # Split and scale data for the current fold
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # Scale data
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)
    
    # Train the model
    svc.fit(X_train_fold, y_train_fold)
    
    # Predict probabilities
    y_proba = svc.predict_proba(X_val_fold)[:, 1]
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val_fold, y_proba)
    tprs.append(np.interp(mean_fpr, fpr, tpr))  # Interpolate for mean ROC
    tprs[-1][0] = 0.0  # Ensure the curve starts at 0
    roc_aucs.append(auc(fpr, tpr))
    
    # Predict labels
    y_pred = svc.predict(X_val_fold)
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_val_fold, y_pred))
    
    # Compute and store classification report
    report = classification_report(y_val_fold, y_pred, output_dict=True)
    classification_reports.append(report)

mean_accuracy = np.mean([report['accuracy'] for report in classification_reports])
print(f"\nMean Accuracy Across 5-Fold CV: {mean_accuracy:.4f}")

# Average ROC curve and AUC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  # Ensure it ends at (1,1)
mean_auc = auc(mean_fpr, mean_tpr)

# Aggregate confusion matrix
avg_conf_matrix = np.mean(conf_matrices, axis=0)
avg_conf_matrix = np.round(avg_conf_matrix).astype(int)

# Aggregate classification report
classification_dfs = [pd.DataFrame(report).transpose() for report in classification_reports]

# Concatenate all classification reports into one DataFrame
combined_classification_df = pd.concat(classification_dfs, axis=0)

# Compute the mean for numeric metrics only
avg_classification_report = combined_classification_df.mean(numeric_only=True)



# Print Average Classification Report (Cross-Validation)
print("\nAverage Classification Report Across 5-Fold CV:")
print(avg_classification_report)


# Final model evaluation on the hold-out test set (not used in cross-validation)
X_train_scaled = scaler.fit_transform(X_train)  # Scale the entire training data
X_test_scaled = scaler.transform(X_test)  # Scale the final test set using the same scaler

# Train the final model on the entire training set
svc.fit(X_train_scaled, y_train)

# Predict on the test set
y_test_pred = svc.predict(X_test_scaled)

# Compute and print final test set metrics
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred)

# Compute ROC for the test set
y_test_proba = svc.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
test_auc = auc(fpr, tpr)

# Print Final Evaluation Metrics
print("\nFinal Test Set Evaluation:")
print("\nTest Set Classification Report:")
print(test_report)

# Final Test Set Confusion Matrix
print("\nTest Set Confusion Matrix:")
print(test_conf_matrix)

# End timing
end_time = time.time()
print(f"\nExecution Time: {end_time - start_time:.2f} seconds")

# Plot the final test set ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label=f'Test Set ROC (AUC = {math.floor(test_auc * 100) / 100})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Set ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Phishing", "Phishing"], yticklabels=["Not Phishing", "Phishing"])
plt.title("Confusion Matrix Testing (SVM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


