import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline

from Enums.ParamGrids import ParamGrids

from BaseStudentPerformance import BaseStudentPerformance

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)
import scipy.stats as stats


class Regression(BaseStudentPerformance):
    def __init__(self, target="G3"):
        super().__init__()
        self.transform_target_regression(target)
        self.target_column = target

    def create_pipeline(self, regressor_name):
        """Create pipeline for regression"""
        regressor = self.get_regressor(regressor_name)

        # Basic pipeline without SMOTE (not applicable to regression)
        pipeline = ImbPipeline(
            [
                ("binary_encode", self.binary_transformer),
                ("column_transform", self.column_transformer),
                ("regressor", regressor),
            ]
        )

        return pipeline

    def perform_grid_search(
        self, X_train, y_train, regressor_name, param_grid=None, cv=5, n_jobs=-1
    ):
        """Perform grid search for regression"""
        # Get the parameter grid
        if param_grid is None:
            param_grid_enum = getattr(
                ParamGrids, f"{regressor_name.upper()}_REGRESSION", None
            )
            if param_grid_enum is None:
                raise ValueError(f"No parameter grid found for {regressor_name}")
            param_grid = param_grid_enum.value

        # Create pipeline
        pipeline = self.create_pipeline(regressor_name)

        # Use multiple scoring metrics for regression
        scoring = {
            "neg_mse": "neg_mean_squared_error",
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
            "r2": "r2",
            "neg_mape": "neg_mean_absolute_percentage_error",
        }

        # For regression, use R2 as the primary metric
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit="r2",  # Choose best model based on R2
            n_jobs=n_jobs,
            verbose=1,
        )

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Store results
        self.grid_search_results = {
            "regressor": regressor_name,
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
        """Evaluate the model for regression"""
        # Use the best model from grid search
        if grid_search is None:
            model = self.grid_search_results["best_model"]
        else:
            model = grid_search.best_estimator_

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate comprehensive metrics
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "explained_variance": explained_variance_score(y_test, y_pred),
            "predictions": y_pred,
            "y_test": y_test,
            "residuals": y_test - y_pred,
        }

        # Add statistical tests
        # Normality test on residuals
        _, p_value_normality = stats.normaltest(metrics["residuals"])
        metrics["residuals_normal_p_value"] = p_value_normality

        # Homoscedasticity test (Breusch-Pagan)
        # This is simplified and can be replaced with proper test
        metrics["residuals_std"] = np.std(metrics["residuals"])

        self.evaluation_results = metrics
        return metrics

    def plot_results(self, figsize=(16, 12)):
        """Enhanced plotting for regression"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)

        y_test = self.evaluation_results["y_test"]
        y_pred = self.evaluation_results["predictions"]
        residuals = self.evaluation_results["residuals"]

        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        perfect_line = np.linspace(y_test.min(), y_test.max(), 100)
        axes[0, 0].plot(
            perfect_line, perfect_line, "r--", lw=2, label="Perfect Prediction"
        )
        axes[0, 0].set_xlabel("Actual Values")
        axes[0, 0].set_ylabel("Predicted Values")
        axes[0, 0].set_title("Actual vs Predicted Values")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Add R² annotation
        r2 = self.evaluation_results["r2"]
        axes[0, 0].text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=axes[0, 0].transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 2. Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0, 1].set_xlabel("Predicted Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Predicted Values")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals Histogram with Normal Distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor="black", density=True)

        # Fit normal distribution to residuals
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 0].plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            "r-",
            lw=2,
            label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})",
        )

        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Distribution of Residuals")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Add normality test p-value
        p_value_normality = self.evaluation_results["residuals_normal_p_value"]
        axes[1, 0].text(
            0.05,
            0.95,
            f"Normality test p-value: {p_value_normality:.4f}",
            transform=axes[1, 0].transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 4. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Normal Q-Q Plot of Residuals")
        axes[1, 1].grid(True, alpha=0.3)

        # 5. Metrics Summary
        metrics_text = f"""
        Regression Metrics:
        
        R² Score: {self.evaluation_results["r2"]:.4f}
        Mean Squared Error: {self.evaluation_results["mse"]:.4f}
        Root Mean Squared Error: {self.evaluation_results["rmse"]:.4f}
        Mean Absolute Error: {self.evaluation_results["mae"]:.4f}
        Mean Absolute Percentage Error: {self.evaluation_results["mape"] * 100:.2f}%
        Explained Variance: {self.evaluation_results["explained_variance"]:.4f}
        
        Best Cross-validation Scores:
        """

        for metric, score in self.grid_search_results["best_scores"].items():
            # Convert negative scores to positive for display
            if metric.startswith("neg_"):
                score = -score
                metric_name = metric[4:].upper()
            else:
                metric_name = metric.upper()
            metrics_text += f"\n{metric_name}: {score:.4f}"

        axes[2, 0].text(
            0.05,
            0.95,
            metrics_text,
            fontsize=11,
            verticalalignment="top",
            transform=axes[2, 0].transAxes,
        )
        axes[2, 0].axis("off")

        # 6. Feature Importance
        best_model = self.grid_search_results["best_model"]
        if hasattr(best_model.named_steps["regressor"], "feature_importances_"):
            importances = best_model.named_steps["regressor"].feature_importances_
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

    def plot_residual_analysis(self, figsize=(15, 10)):
        """Detailed residual analysis plots"""
        residuals = self.evaluation_results["residuals"]
        y_test = self.evaluation_results["y_test"]
        y_pred = self.evaluation_results["predictions"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Residuals vs Index (to check for patterns)
        axes[0, 0].scatter(range(len(residuals)), residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0, 0].set_xlabel("Observation Index")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Index")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Scale-Location Plot (sqrt of standardized residuals vs fitted)
        standardized_residuals = residuals / np.std(residuals)
        sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))

        axes[0, 1].scatter(y_pred, sqrt_std_residuals, alpha=0.6)
        axes[0, 1].set_xlabel("Predicted Values")
        axes[0, 1].set_ylabel("√|Standardized Residuals|")
        axes[0, 1].set_title("Scale-Location Plot")
        axes[0, 1].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(y_pred, sqrt_std_residuals, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(sorted(y_pred), p(sorted(y_pred)), "r--", alpha=0.8)

        # 3. Residuals vs Actual Values
        axes[1, 0].scatter(y_test, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[1, 0].set_xlabel("Actual Values")
        axes[1, 0].set_ylabel("Residuals")
        axes[1, 0].set_title("Residuals vs Actual Values")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Box plot of residuals
        axes[1, 1].boxplot(residuals, vert=True)
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Box Plot of Residuals")
        axes[1, 1].grid(True, alpha=0.3)

        # Add outlier information
        Q1 = np.percentile(residuals, 25)
        Q3 = np.percentile(residuals, 75)
        IQR = Q3 - Q1
        outliers = residuals[
            (residuals < Q1 - 1.5 * IQR) | (residuals > Q3 + 1.5 * IQR)
        ]
        axes[1, 1].text(
            0.95,
            0.95,
            f"Outliers: {len(outliers)}",
            transform=axes[1, 1].transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()

    def plot_prediction_error_distribution(self, figsize=(12, 8)):
        """Plot the distribution of prediction errors"""
        y_test = self.evaluation_results["y_test"]
        y_pred = self.evaluation_results["predictions"]

        # Calculate percentage errors, handling zero values
        # Only calculate percentage errors for non-zero actual values
        non_zero_mask = y_test != 0
        if non_zero_mask.sum() > 0:
            percentage_errors = np.where(
                non_zero_mask, ((y_pred - y_test) / y_test) * 100, np.nan
            )
            # Remove NaN values for plotting
            percentage_errors_clean = percentage_errors[~np.isnan(percentage_errors)]
        else:
            percentage_errors_clean = np.array([])

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Absolute Error Distribution
        abs_errors = np.abs(y_pred - y_test)
        axes[0, 0].hist(abs_errors, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 0].axvline(
            x=np.mean(abs_errors),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(abs_errors):.2f}",
        )
        axes[0, 0].axvline(
            x=np.median(abs_errors),
            color="g",
            linestyle="--",
            label=f"Median: {np.median(abs_errors):.2f}",
        )
        axes[0, 0].set_xlabel("Absolute Error")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Distribution of Absolute Errors")
        axes[0, 0].legend()

        # 2. Percentage Error Distribution (only for non-zero actuals)
        if len(percentage_errors_clean) > 0:
            # Remove extreme outliers for better visualization
            Q1 = np.percentile(percentage_errors_clean, 25)
            Q3 = np.percentile(percentage_errors_clean, 75)
            IQR = Q3 - Q1
            outlier_mask = (percentage_errors_clean >= Q1 - 1.5 * IQR) & (
                percentage_errors_clean <= Q3 + 1.5 * IQR
            )
            percentage_errors_filtered = percentage_errors_clean[outlier_mask]

            axes[0, 1].hist(
                percentage_errors_filtered, bins=30, alpha=0.7, edgecolor="black"
            )
            axes[0, 1].axvline(x=0, color="r", linestyle="-", lw=2)
            if len(percentage_errors_filtered) > 0:
                axes[0, 1].axvline(
                    x=np.mean(percentage_errors_filtered),
                    color="g",
                    linestyle="--",
                    label=f"Mean: {np.mean(percentage_errors_filtered):.2f}%",
                )
            axes[0, 1].set_xlabel("Percentage Error (%)")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title(
                f"Distribution of Percentage Errors\n(Excluding {np.sum(~outlier_mask)} outliers)"
            )
            axes[0, 1].legend()

            # Add note about zero values if any
            n_zeros = np.sum(y_test == 0)
            if n_zeros > 0:
                axes[0, 1].text(
                    0.95,
                    0.95,
                    f"Note: {n_zeros} samples with actual=0 excluded",
                    transform=axes[0, 1].transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                )
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No valid percentage errors\n(all actual values are 0)",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Distribution of Percentage Errors")

        # 3. Error vs Magnitude
        axes[1, 0].scatter(y_test, abs_errors, alpha=0.6)
        axes[1, 0].set_xlabel("Actual Values")
        axes[1, 0].set_ylabel("Absolute Error")
        axes[1, 0].set_title("Error Magnitude vs Actual Values")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Cumulative Error Distribution
        sorted_abs_errors = np.sort(abs_errors)
        cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
        axes[1, 1].plot(sorted_abs_errors, cumulative)
        axes[1, 1].set_xlabel("Absolute Error")
        axes[1, 1].set_ylabel("Cumulative Probability")
        axes[1, 1].set_title("Cumulative Distribution of Absolute Errors")
        axes[1, 1].grid(True, alpha=0.3)

        # Add percentile lines
        percentiles = [50, 75, 90, 95]
        for p in percentiles:
            val = np.percentile(abs_errors, p)
            axes[1, 1].axvline(x=val, color="r", linestyle="--", alpha=0.5)
            axes[1, 1].text(val, p / 100, f"{p}%", rotation=90, va="bottom")

        plt.tight_layout()
        plt.show()

    def analyze_prediction_intervals(self, X_test, confidence=0.95):
        """Create prediction intervals using bootstrap"""
        from sklearn.utils import resample

        n_bootstraps = 100
        y_pred_matrix = np.zeros((len(X_test), n_bootstraps))

        # Bootstrap predictions
        for i in range(n_bootstraps):
            # Create bootstrap sample
            X_train_boot, y_train_boot = resample(self.X, self.y, random_state=i)

            # Train model on bootstrap sample
            boot_model = self.grid_search_results["best_model"]
            boot_model.fit(X_train_boot, y_train_boot)

            # Make predictions
            y_pred_matrix[:, i] = boot_model.predict(X_test)

        # Calculate prediction intervals
        alpha = (1 - confidence) / 2
        percentiles = [alpha * 100, (1 - alpha) * 100]

        y_pred_mean = np.mean(y_pred_matrix, axis=1)
        y_pred_lower = np.percentile(y_pred_matrix, percentiles[0], axis=1)
        y_pred_upper = np.percentile(y_pred_matrix, percentiles[1], axis=1)

        # Plot prediction intervals
        fig, ax = plt.subplots(figsize=(12, 8))

        # Sort by mean prediction for better visualization
        sort_idx = np.argsort(y_pred_mean)

        x_plot = range(len(y_pred_mean))
        ax.plot(x_plot, y_pred_mean[sort_idx], "b-", label="Predicted")
        ax.fill_between(
            x_plot,
            y_pred_lower[sort_idx],
            y_pred_upper[sort_idx],
            alpha=0.3,
            color="blue",
            label=f"{int(confidence * 100)}% Prediction Interval",
        )

        # Add actual values - Convert Series to numpy array for proper indexing
        y_test = self.evaluation_results["y_test"]
        if isinstance(y_test, pd.Series):
            y_test_array = y_test.values
        else:
            y_test_array = y_test

        ax.scatter(
            x_plot, y_test_array[sort_idx], color="red", alpha=0.6, s=10, label="Actual"
        )

        ax.set_xlabel("Sample Index (sorted by prediction)")
        ax.set_ylabel("Target Value")
        ax.set_title(f"Prediction Intervals ({int(confidence * 100)}% confidence)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate coverage
        coverage = np.mean(
            (y_test_array >= y_pred_lower) & (y_test_array <= y_pred_upper)
        )
        print(f"Actual coverage: {coverage:.3f} (expected: {confidence:.3f})")

        return {
            "y_pred_mean": y_pred_mean,
            "y_pred_lower": y_pred_lower,
            "y_pred_upper": y_pred_upper,
            "coverage": coverage,
        }
