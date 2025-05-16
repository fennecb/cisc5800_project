import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_confusion_matrix(precision, recall, test_positives, test_negatives):
    """
    Calculate confusion matrix from precision and recall for binary classification.
    
    In this context:
    - Positive class = Failing students (class 0)
    - Negative class = Passing students (class 1)
    
    Parameters:
    -----------
    precision: Precision for the positive class (failing students)
    recall: Recall for the positive class (failing students)
    test_positives: Number of actual positive cases in test set
    test_negatives: Number of actual negative cases in test set
    
    Returns:
    --------
    2x2 confusion matrix as numpy array in the format:
    [[TN, FP],
     [FN, TP]]
    """
    # Calculate TP (True Positives): Correctly identified failing students
    TP = round(recall * test_positives)
    
    # Calculate FN (False Negatives): Failing students incorrectly identified as passing
    FN = test_positives - TP
    
    # Calculate FP (False Positives): Passing students incorrectly identified as failing
    # Using precision: precision = TP / (TP + FP)
    if precision > 0:
        # Rearranging: FP = TP/precision - TP = TP * (1/precision - 1)
        FP = round(TP / precision - TP)
    else:
        FP = 0
    
    # Calculate TN (True Negatives): Passing students correctly identified as passing
    TN = test_negatives - FP
    
    # Create and return confusion matrix
    cm = np.array([[TN, FP], [FN, TP]])
    
    return cm

# Get file path from user input
file_path = input("Enter the path to the JSON file: ")

# Validate if the file exists
while not os.path.isfile(file_path):
    print(f"Error: File '{file_path}' does not exist.")
    file_path = input("Please enter a valid file path: ")

# Load the data
try:
    with open(file_path, 'r') as f:
        results = json.load(f)
except json.JSONDecodeError:
    print("Error: Invalid JSON format.")
    exit(1)
except Exception as e:
    print(f"Error opening file: {e}")
    exit(1)

# Check if binary results exist
if "binary" not in results:
    print("Error: The JSON file does not contain 'binary' results.")
    exit(1)

binary_results = results["binary"]

# Test set distribution (from the original dataset with test_size=0.33)
# Class 0 = failing students (minority class)
# Class 1 = passing students (majority class)
total_positives = 100  # Total class 0 (fail) in full dataset
total_negatives = 549  # Total class 1 (pass) in full dataset
test_size = 0.33

# Calculate test set size for each class, ensuring they're integers
test_positives = round(total_positives * test_size)
test_negatives = round(total_negatives * test_size)

print(f"Estimated test set: {test_positives + test_negatives} samples")
print(f"  - Class 0 (failing): {test_positives} samples")
print(f"  - Class 1 (passing): {test_negatives} samples")

# Create subplot grid that can fit all models
models = list(binary_results.keys())
n_models = len(models)
n_cols = min(3, n_models)
n_rows = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

# Convert to 1D array if there's only one row
if n_rows == 1:
    axes = np.array([axes])
if n_cols == 1:
    axes = axes.reshape(-1, 1)

# Flatten for easier iteration
axes_flat = axes.flatten()

# Plot confusion matrices for each model
for i, model_name in enumerate(models):
    # Skip if model doesn't exist in results
    if model_name not in binary_results:
        continue
    
    metrics = binary_results[model_name]
    
    # Extract metrics (handle potential missing keys)
    precision = metrics.get("minority_class_precision", 0)
    recall = metrics.get("minority_class_recall", 0)
    
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(precision, recall, test_positives, test_negatives)
    
    # Calculate row percentages for color intensity
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percentages = np.zeros_like(cm, dtype=float)
    for j in range(cm.shape[0]):
        if row_sums[j] > 0:
            cm_percentages[j] = cm[j] / row_sums[j]
    
    # Plot with seaborn
    sns.heatmap(cm_percentages, annot=cm, fmt='d', cmap='Blues', ax=axes_flat[i],
                cbar=False, annot_kws={"size": 12})
    
    # Add labels
    axes_flat[i].set_xlabel('Predicted', fontsize=12)
    axes_flat[i].set_ylabel('Actual', fontsize=12)
    axes_flat[i].set_title(f'{model_name.replace("_", " ").title()}', fontsize=14)
    
    # Set x and y ticks (bottom-left is TN)
    axes_flat[i].set_xticklabels(['Pass (1)', 'Fail (0)'])
    axes_flat[i].set_yticklabels(['Pass (1)', 'Fail (0)'])

# Hide empty subplots if any
for j in range(len(models), len(axes_flat)):
    axes_flat[j].axis('off')

plt.tight_layout()
plt.suptitle('Confusion Matrices for Different Models (Binary Classification)', fontsize=16, y=1.05)

# Ask if user wants to save the plot
save_plot = input("Do you want to save the plot? (y/n): ").lower()
if save_plot == 'y' or save_plot == 'yes':
    output_path = input("Enter the path to save the plot: ")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

plt.show()