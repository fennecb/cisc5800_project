from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
    roc_curve,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from itertools import cycle

from Enums.ParamGrids import ParamGrids
from BaseStudentPerformance import BaseStudentPerformance


class MultiClassification(BaseStudentPerformance):
    def __init__(self, n_bins=5, strategy="quantile"):
        super().__init__()
        self.transform_target_multiclass(n_bins, strategy)
        self.n_bins = n_bins
        self.strategy = strategy

    def create_pipeline(self, classifier_name):
        """Create pipeline for multi-class classification"""
        classifier = self.get_classifier(classifier_name)

        # For multi-class, class_weight='balanced' handles all classes
        if hasattr(classifier, "class_weight"):
            classifier.set_params(class_weight="balanced")

        # Note: SMOTE is typically not used with multi-class
        pipeline = ImbPipeline(
            [
                ("binary_encode", self.binary_transformer),
                ("column_transform", self.column_transformer),
                ("classifier", classifier),
            ]
        )

        return pipeline

    def perform_grid_search(
        self, X_train, y_train, classifier_name, param_grid=None, cv=5, n_jobs=-1
    ):
        """Perform grid search for multi-class classification"""
        # Get the parameter grid
        if param_grid is None:
            param_grid_enum = getattr(
                ParamGrids, f"{classifier_name.upper()}_MULTICLASS", None
            )
            if param_grid_enum is None:
                # Fallback to binary grid if multi-class specific not found
                param_grid_enum = getattr(
                    ParamGrids, f"{classifier_name.upper()}_BINARY", None
                )
            if param_grid_enum is None:
                raise ValueError(f"No parameter grid found for {classifier_name}")
            param_grid = param_grid_enum.value

        # Create pipeline
        pipeline = self.create_pipeline(classifier_name)

        # Use multiple scoring metrics for multi-class
        scoring = {
            "precision_macro": make_scorer(precision_score, average="macro"),
            "recall_macro": make_scorer(recall_score, average="macro"),
            "f1_macro": make_scorer(f1_score, average="macro"),
            "f1_weighted": make_scorer(f1_score, average="weighted"),
            "accuracy": "accuracy",
        }

        # For multi-class, use macro F1 as the primary metric
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit="f1_macro",  # Choose best model based on macro F1
            n_jobs=n_jobs,
            verbose=1,
        )

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Store results
        self.grid_search_results = {
            "classifier": classifier_name,
            "best_params": grid_search.best_params_,
            "best_scores": {
                metric: grid_search.cv_results_[f"mean_test_{metric}"][
                    grid_search.best_index_
                ]
                for metric in scoring.keys()
            },
            "cv_results": grid_search.cv_results_,
            "best_model": grid_search.best_estimator_,
        }

        # Store fitted pipeline for feature names
        self.fitted_pipeline = grid_search.best_estimator_

        return grid_search

    def evaluate_model(self, X_test, y_test, grid_search=None):
        """Evaluate the model for multi-class classification"""
        # Use the best model from grid search
        if grid_search is None:
            model = self.grid_search_results["best_model"]
        else:
            model = grid_search.best_estimator_

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate comprehensive metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "predictions": y_pred,
            "prediction_probabilities": y_pred_proba,
            "y_test": y_test,  # Store for plotting
            "classes": sorted(np.unique(y_test)),
        }

        # Per-class metrics
        class_precision = precision_score(y_test, y_pred, average=None)
        class_recall = recall_score(y_test, y_pred, average=None)
        class_f1 = f1_score(y_test, y_pred, average=None)

        metrics["per_class_precision"] = dict(zip(metrics["classes"], class_precision))
        metrics["per_class_recall"] = dict(zip(metrics["classes"], class_recall))
        metrics["per_class_f1"] = dict(zip(metrics["classes"], class_f1))

        self.evaluation_results = metrics
        return metrics

    def plot_results(self, figsize=(16, 12)):
        """Enhanced plotting for multi-class classification"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # 1. Confusion Matrix
        cm = self.evaluation_results["confusion_matrix"]
        classes = self.evaluation_results["classes"]

        # Create percentage confusion matrix
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        im = axes[0, 0].imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
        axes[0, 0].set_title("Confusion Matrix (%)")

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0, 0])
        cbar.ax.set_ylabel("Percentage", rotation=-90, va="bottom")

        # Add labels
        tick_marks = np.arange(len(classes))
        axes[0, 0].set_xticks(tick_marks)
        axes[0, 0].set_yticks(tick_marks)
        axes[0, 0].set_xticklabels(classes)
        axes[0, 0].set_yticklabels(classes)

        # Add text annotations
        for i, j in np.ndindex(cm_percent.shape):
            axes[0, 0].text(
                j,
                i,
                f"{cm_percent[i, j]:.1f}%",
                horizontalalignment="center",
                color="white" if cm_percent[i, j] > 50 else "black",
            )

        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")
        axes[0, 0].set_xticklabels(classes, rotation=45)

        # 2. Per-class F1 scores
        f1_scores = list(self.evaluation_results["per_class_f1"].values())
        axes[0, 1].bar(range(len(classes)), f1_scores)
        axes[0, 1].set_xticks(range(len(classes)))
        axes[0, 1].set_xticklabels(classes)
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].set_title("Per-Class F1 Scores")
        axes[0, 1].set_ylim(0, 1)

        # Add value labels on bars
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f"{v:.3f}", ha="center")

        # 3. MultiClass ROC Curve
        y_test = self.evaluation_results["y_test"]
        y_pred_proba = self.evaluation_results["prediction_probabilities"]

        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = len(classes)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red"])
        for i, color in zip(range(n_classes), colors):
            axes[1, 0].plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})",
            )

        axes[1, 0].plot([0, 1], [0, 1], "k--", lw=2)
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("Multi-class ROC Curves")
        axes[1, 0].legend(loc="lower right")

        # 4. Class Distribution
        class_counts = pd.Series(y_test).value_counts().sort_index()
        axes[1, 1].bar(range(len(class_counts)), class_counts.values)
        axes[1, 1].set_xticks(range(len(class_counts)))
        axes[1, 1].set_xticklabels(class_counts.index)
        axes[1, 1].set_xlabel("Class")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Class Distribution")

        # Add value labels on bars
        for i, v in enumerate(class_counts.values):
            axes[1, 1].text(i, v + 0.5, str(v), ha="center")

        # 5. Metrics Summary
        metrics_text = f"""
        Accuracy: {self.evaluation_results["accuracy"]:.3f}
        F1 Macro: {self.evaluation_results["f1_macro"]:.3f}
        F1 Weighted: {self.evaluation_results["f1_weighted"]:.3f}
        Precision Macro: {self.evaluation_results["precision_macro"]:.3f}
        Recall Macro: {self.evaluation_results["recall_macro"]:.3f}
        
        Best Cross-validation Scores:
        """

        for metric, score in self.grid_search_results["best_scores"].items():
            metrics_text += f"\n{metric}: {score:.3f}"

        axes[2, 0].text(0.1, 0.1, metrics_text, fontsize=12, verticalalignment="top")
        axes[2, 0].axis("off")

        # 6. Feature Importance with real names
        best_model = self.grid_search_results["best_model"]
        if hasattr(best_model.named_steps["classifier"], "feature_importances_"):
            importances = best_model.named_steps["classifier"].feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features

            # Get real feature names
            feature_names = self.get_feature_names_after_preprocessing()
            feature_names_top = [feature_names[i] for i in indices]

            axes[2, 1].bar(range(len(indices)), importances[indices])
            axes[2, 1].set_xticks(range(len(indices)))
            axes[2, 1].set_xticklabels(feature_names_top, rotation=45, ha="right")
            axes[2, 1].set_title("Top 10 Feature Importances")
            axes[2, 1].set_ylabel("Importance")
        else:
            axes[2, 1].text(
                0.5, 0.5, "Feature importance not available", ha="center", va="center"
            )
            axes[2, 1].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix_detailed(self, figsize=(10, 8)):
        """Plot a more detailed confusion matrix with both counts and percentages"""
        cm = self.evaluation_results["confusion_matrix"]
        classes = self.evaluation_results["classes"]

        fig, ax = plt.subplots(figsize=figsize)

        # Create annotations with both count and percentage
        annot = np.empty_like(cm).astype(str)
        for i in range(len(classes)):
            for j in range(len(classes)):
                count = cm[i, j]
                percent = count / cm[i, :].sum() * 100
                annot[i, j] = f"{count}\n({percent:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Count"},
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Detailed Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_per_class_metrics(self, figsize=(12, 8)):
        """Plot detailed per-class metrics"""
        classes = self.evaluation_results["classes"]
        precision = list(self.evaluation_results["per_class_precision"].values())
        recall = list(self.evaluation_results["per_class_recall"].values())
        f1 = list(self.evaluation_results["per_class_f1"].values())

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width, precision, width, label="Precision")
        bars2 = ax.bar(x, recall, width, label="Recall")
        bars3 = ax.bar(x + width, f1, width, label="F1 Score")

        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Performance Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        plt.tight_layout()
        plt.show()

    def analyze_misclassifications(self, X_test):
        """Analyze misclassified samples by looking at confusion patterns"""
        y_test = self.evaluation_results["y_test"]
        y_pred = self.evaluation_results["predictions"]
        classes = self.evaluation_results["classes"]

        # Create a DataFrame for analysis
        misclassified_analysis = []

        for true_class in classes:
            for pred_class in classes:
                if true_class != pred_class:
                    mask = (y_test == true_class) & (y_pred == pred_class)
                    count = mask.sum()
                    if count > 0:
                        misclassified_analysis.append(
                            {
                                "true_class": true_class,
                                "predicted_class": pred_class,
                                "count": count,
                                "percentage": count
                                / (y_test == true_class).sum()
                                * 100,
                            }
                        )

        misclassified_df = pd.DataFrame(misclassified_analysis)

        # Plot misclassification patterns
        if len(misclassified_df) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

            # Bar chart of misclassifications
            misclassified_df["label"] = misclassified_df.apply(
                lambda x: f"{x['true_class']} → {x['predicted_class']}", axis=1
            )
            misclassified_df = misclassified_df.sort_values("count", ascending=False)

            ax1.bar(misclassified_df["label"], misclassified_df["count"])
            ax1.set_xlabel("True Class → Predicted Class")
            ax1.set_ylabel("Count")
            ax1.set_title("Misclassification Patterns by Count")
            ax1.tick_params(axis="x", rotation=45)

            # Percentage chart
            ax2.bar(misclassified_df["label"], misclassified_df["percentage"])
            ax2.set_xlabel("True Class → Predicted Class")
            ax2.set_ylabel("Percentage of True Class")
            ax2.set_title("Misclassification Patterns by Percentage")
            ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.show()

        return misclassified_df

    def plot_prediction_confidence(self, figsize=(12, 8)):
        """Plot the confidence of predictions for each class"""
        y_test = self.evaluation_results["y_test"]
        y_pred_proba = self.evaluation_results["prediction_probabilities"]
        classes = self.evaluation_results["classes"]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()

        for idx, class_label in enumerate(classes):
            if idx < len(axes):
                # Get max probability for predictions of this class
                class_mask = y_test == class_label
                if class_mask.sum() > 0:
                    class_confidences = y_pred_proba[class_mask, idx]
                    axes[idx].hist(
                        class_confidences, bins=20, alpha=0.7, edgecolor="black"
                    )
                    axes[idx].axvline(
                        x=1 / len(classes),
                        color="red",
                        linestyle="--",
                        label=f"Random ({1 / len(classes):.3f})",
                    )
                    axes[idx].set_title(
                        f"Prediction Confidence for Class {class_label}"
                    )
                    axes[idx].set_xlabel("Probability")
                    axes[idx].set_ylabel("Count")
                    axes[idx].legend()

        # Hide unused subplots
        for idx in range(len(classes), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()
