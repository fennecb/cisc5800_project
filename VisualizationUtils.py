def plot_model_comparison(models_results, output_path="model_comparison_chart.png"):
    """
    Create a bar chart comparing multiple models across key performance metrics.

    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and dictionaries of metrics as values.
        Example: {'Random Forest': {'accuracy': 0.85, 'f1_score': 0.75, ...}, ...}
    output_path : str, optional
        Path to save the output figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(12, 8))

    # Select the metrics to display
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
    ]
    metrics_display = [
        "Accuracy",
        "Balanced Accuracy",
        "F1 Score",
        "Precision",
        "Recall",
        "ROC AUC",
    ]

    # Prepare the data for plotting
    data = {}
    for model_name, results in models_results.items():
        # Convert keys to lowercase for consistency
        results_lower = {k.lower(): v for k, v in results.items()}
        data[model_name] = [results_lower.get(metric, 0) for metric in metrics]

    # Create DataFrame for easier plotting
    df_results = pd.DataFrame(data, index=metrics_display)

    # Plot
    ax = df_results.plot(kind="bar", rot=0, colormap="viridis")
    plt.title("Performance Comparison of Classification Models", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return ax


def plot_precision_recall_curves(
    models_proba, y_test, output_path="precision_recall_curves.png"
):
    """
    Create precision-recall curves for multiple models.

    Parameters:
    -----------
    models_proba : dict
        Dictionary with model names as keys and arrays of predicted probabilities as values.
        Example: {'Random Forest': np.array([0.1, 0.8, 0.3, ...]), ...}
    y_test : array-like
        True labels for the test set
    output_path : str, optional
        Path to save the output figure
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(10, 8))

    for name, y_prob in models_proba.items():
        # For identifying failing students (assuming class 0 is "fail")
        # We need to use 1-y_prob because predict_proba returns P(class=1)
        # and we want P(class=0) for the failing class
        precision, recall, _ = precision_recall_curve(y_test, 1 - y_prob, pos_label=0)
        avg_precision = average_precision_score(y_test, 1 - y_prob, pos_label=0)
        plt.plot(recall, precision, lw=2, label=f"{name} (AP = {avg_precision:.2f})")

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall Curves for Identifying Failing Students", fontsize=16)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plt


def plot_feature_importance(
    model, feature_names, top_n=10, output_path="feature_importance.png"
):
    """
    Create a bar chart showing the most important features for a model.

    Parameters:
    -----------
    model : fitted model object
        The trained model with feature_importances_ attribute (e.g., RandomForest)
    feature_names : list
        List of feature names corresponding to the training data columns
    top_n : int, optional
        Number of top features to display
    output_path : str, optional
        Path to save the output figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    plt.figure(figsize=(12, 8))

    # Extract feature importances
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")

    # Get top N features
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]

    # Plot
    sns.barplot(x=importance[indices], y=top_features)
    plt.title(f"Top {top_n} Most Important Features", fontsize=16)
    plt.xlabel("Relative Importance", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plt


def plot_probability_distribution(
    y_proba,
    y_test,
    threshold=0.5,
    optimal_threshold=None,
    output_path="probability_distribution.png",
):
    """
    Create histograms showing the distribution of predicted probabilities by actual class.

    Parameters:
    -----------
    y_proba : array-like
        Predicted probabilities from a model (probability of class 1)
    y_test : array-like
        True labels for the test set
    threshold : float, optional
        Default classification threshold to mark
    optimal_threshold : float, optional
        Optimized threshold to mark (if applicable)
    output_path : str, optional
        Path to save the output figure
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Split probabilities by actual class
    # For failing students (class 0), we're plotting their probability of passing (class 1)
    prob_failing = y_proba[y_test == 0]
    prob_passing = y_proba[y_test == 1]

    # Plot histograms
    plt.hist(
        prob_failing,
        bins=20,
        alpha=0.6,
        color="red",
        label=f"Failing Students (n={len(prob_failing)})",
    )
    plt.hist(
        prob_passing,
        bins=20,
        alpha=0.6,
        color="green",
        label=f"Passing Students (n={len(prob_passing)})",
    )

    # Add threshold lines
    plt.axvline(
        x=threshold,
        color="black",
        linestyle="--",
        label=f"Default Threshold ({threshold:.2f})",
    )

    if optimal_threshold is not None:
        plt.axvline(
            x=optimal_threshold,
            color="blue",
            linestyle=":",
            label=f"Optimized Threshold ({optimal_threshold:.2f})",
        )

    plt.xlabel("Predicted Probability of Passing", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Probability Distribution by Actual Class", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plt


def plot_threshold_analysis(
    y_proba, y_test, thresholds=None, output_path="threshold_analysis.png"
):
    """
    Create a plot showing how precision, recall, and F1 change with different thresholds.

    Parameters:
    -----------
    y_proba : array-like
        Predicted probabilities from a model (probability of class 1)
    y_test : array-like
        True labels for the test set
    thresholds : array-like, optional
        Array of threshold values to evaluate. If None, creates a range from 0.01 to 0.99
    output_path : str, optional
        Path to save the output figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        # Convert probabilities to binary predictions based on threshold
        # Remember to convert to class 0 (1 - y_proba >= threshold)
        y_pred = (y_proba < threshold).astype(
            int
        )  # If y_proba < threshold, predict class 0 (fail)

        # Calculate metrics for failing class (class 0)
        try:
            precision = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        except:
            # Skip thresholds that result in invalid calculations
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)

    # Find optimal threshold for F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(
        thresholds, precisions, "b-", label="Precision (Failing Students)", linewidth=2
    )
    plt.plot(thresholds, recalls, "g-", label="Recall (Failing Students)", linewidth=2)
    plt.plot(
        thresholds, f1_scores, "r-", label="F1 Score (Failing Students)", linewidth=2
    )

    plt.axvline(x=0.5, color="black", linestyle="--", label="Default Threshold (0.5)")
    plt.axvline(
        x=optimal_threshold,
        color="red",
        linestyle=":",
        label=f"Optimal F1 Threshold ({optimal_threshold:.2f})",
    )

    plt.xlabel('Probability Threshold for Predicting "Pass" (Class 1)', fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title(
        "Performance Metrics vs Classification Threshold\n(All metrics calculated for identifying failing students)",
        fontsize=16,
    )
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    # Add annotation explaining the threshold interpretation
    plt.text(
        0.02,
        0.05,
        "Lower threshold → More students predicted to fail\nHigher threshold → Fewer students predicted to fail",
        fontsize=12,
        style="italic",
        bbox=dict(facecolor="white", alpha=0.8),
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plt, optimal_threshold
