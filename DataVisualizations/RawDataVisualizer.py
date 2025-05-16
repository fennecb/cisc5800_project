import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
from ucimlrepo import fetch_ucirepo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RawDataVisualizer:
    def __init__(self):
        try:  # Source the data locally first (faster).
            filename = "student-por.csv"
            file_path = os.path.join(os.getcwd(), filename)
            self.data = pd.read_csv(file_path, sep=";")
            self.X = self.data.iloc[:, :-3]
            self.raw_targets = self.data.iloc[:, -3:]
        except:  # If not there, download from source
            self.data = fetch_ucirepo(id=320)
            self.X = self.data.data.features
            self.raw_targets = self.data.data.targets

    def feature_distribution_visualization(self):
        """
        Creates visualizations showing the distribution of categorical features in the dataset.
        Uses a normalized dataframe approach for better control of the visualization.
        """
        # Select categorical features
        cat_features = self.X.select_dtypes(include=["object"]).columns.tolist()

        # Create a dataframe to store normalized counts
        distribution_df = pd.DataFrame(index=cat_features)

        # Store original labels for later reference
        feature_labels = {}

        # Calculate normalized counts for each feature
        for feature in cat_features:
            # Get value counts
            counts = self.X[feature].value_counts()
            total = counts.sum()

            # Store the labels for this feature
            feature_labels[feature] = counts.index.tolist()

            # Add normalized counts to dataframe
            for i, (value, count) in enumerate(counts.items()):
                # Use positional column names (0, 1, 2, etc.)
                column_name = f"value_{i}"
                distribution_df.loc[feature, column_name] = count / total

        # Fill NaN values with 0 (for features with fewer unique values)
        distribution_df = distribution_df.fillna(0)

        # Sort features by number of unique values for better visualization
        distribution_df["num_values"] = distribution_df.count(axis=1)
        distribution_df = distribution_df.sort_values("num_values")
        distribution_df = distribution_df.drop("num_values", axis=1)

        # Now visualize the data
        plt.figure(figsize=(14, 10))

        # Convert to percentage
        plot_data = distribution_df * 100

        # Create horizontal stacked bar chart
        ax = plot_data.plot(
            kind="barh", stacked=True, figsize=(14, 10), colormap="viridis", width=0.7
        )

        # Add annotations to each segment
        for container in ax.containers:
            # Only annotate segments that are large enough
            for rect in container:
                width = rect.get_width()
                if width > 5:  # Only annotate segments wider than 5%
                    # Get the feature name from y position
                    feature_idx = int(rect.get_y())
                    feature_name = plot_data.index[feature_idx]

                    # Get the value position from the rectangle position in the container
                    value_idx = ax.containers.index(container)

                    # Only add label if this feature has this value
                    if value_idx < len(feature_labels[feature_name]):
                        value_label = feature_labels[feature_name][value_idx]

                        # Add the label and percentage
                        ax.text(
                            rect.get_x() + width / 2,
                            rect.get_y() + rect.get_height() / 2,
                            f"{value_label}\n{width:.1f}%",
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                            fontsize=9,
                        )

        # Improve plot appearance
        plt.title("Distribution of Categorical Features", fontsize=16)
        plt.xlabel("Percentage (%)", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.grid(axis="x", alpha=0.3)
        plt.xlim(0, 100)

        # Remove legend (since we're annotating directly)
        plt.legend().remove()

        # Add imbalance information for binary features
        y_pos = len(cat_features) - 0.5
        for feature in cat_features:
            if len(feature_labels[feature]) == 2:
                # Calculate imbalance ratio
                counts = self.X[feature].value_counts()
                imbalance_ratio = counts.max() / counts.min()

                # Add text
                plt.text(
                    101,
                    y_pos,
                    f"Imbalance: {imbalance_ratio:.2f}:1",
                    va="center",
                    ha="left",
                    fontsize=9,
                    bbox=dict(facecolor="lightyellow", alpha=0.7),
                )

            y_pos -= 1

        plt.tight_layout()
        plt.savefig(
            "DataVisualizations/feature_distribution_horizontal.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create individual bar charts for detailed view
        n_cols = 2
        n_rows = (len(cat_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, feature in enumerate(cat_features):
            # Get value counts
            counts = self.X[feature].value_counts()
            percentages = (counts / counts.sum()) * 100

            # Plot
            ax = axes[i]
            bars = ax.bar(range(len(counts)), percentages, color="skyblue")

            # Add labels and percentages
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(
                counts.index, rotation=45 if len(counts) > 2 else 0, ha="right"
            )

            # Add count and percentage on bars
            for j, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
                height = bar.get_height()
                ax.text(
                    j,
                    height + 1,
                    f"{count}\n({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Add titles and labels
            ax.set_title(f"Distribution of {feature}", fontsize=12)
            ax.set_ylabel("Percentage (%)", fontsize=10)
            ax.set_ylim(0, 100)

            # Add imbalance ratio for binary features
            if len(counts) == 2:
                imbalance_ratio = counts.max() / counts.min()
                ax.text(
                    0.5,
                    0.02,
                    f"Imbalance ratio: {imbalance_ratio:.2f}:1",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(facecolor="lightyellow", alpha=0.5),
                )

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(
            "DataVisualizations/feature_distribution_individual.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def class_distribution_barcharts(self):
        """
        Creates bar charts showing the distribution of key binary and categorical features.
        Highlights class imbalances that may need addressing.
        """
        # Select a subset of interesting categorical features to visualize
        selected_features = [
            "school",
            "sex",
            "address",
            "famsize",
            "Pstatus",
            "schoolsup",
            "famsup",
            "internet",
            "romantic",
        ]

        # Calculate number of rows and columns for subplots
        n_features = len(selected_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

        # Create the plot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(selected_features):
            # Calculate value counts and percentages
            counts = self.X[feature].value_counts()
            percentages = self.X[feature].value_counts(normalize=True) * 100

            # Create the bar chart
            ax = axes[i]
            bars = ax.bar(counts.index, counts.values, color="skyblue")
            ax.set_title(f"Distribution of {feature}", fontsize=12)
            ax.set_ylabel("Count", fontsize=10)

            # Add count and percentage labels on bars
            for bar, count, percentage in zip(bars, counts, percentages):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{count}\n({percentage:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Highlight imbalance ratio
            min_val = counts.min()
            max_val = counts.max()
            imbalance_ratio = max_val / min_val
            ax.text(
                0.5,
                0.02,
                f"Imbalance ratio: {imbalance_ratio:.2f}:1",
                ha="center",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="lightyellow", alpha=0.5),
            )

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(
            "DataVisualizations/class_distribution_barcharts.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def grade_distribution_histogram(self):
        """
        Creates a histogram showing the distribution of final grades (G3).
        Includes overlay for binary classification threshold (pass/fail).
        """
        # Get final grades
        G3 = self.raw_targets["G3"]

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Define bins explicitly to match the grade scale (0-20)
        bins = (
            np.arange(0, 21 + 1) - 0.5
        )  # from -0.5 to 20.5 to center bins on integers

        # Create histogram
        counts, bins, patches = plt.hist(
            G3, bins=bins, alpha=0.7, color="skyblue", edgecolor="black", rwidth=0.8
        )

        # Add count labels directly above each bar
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        max_height = counts.max() * 1.05  # Adjust to ensure labels don't get cut off

        for count, x in zip(counts, bin_centers):
            if count > 0:  # Only label non-empty bins
                plt.text(
                    x,
                    count + max_height * 0.02,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Ensure y-axis extends high enough for labels
        plt.ylim(0, max_height * 1.15)

        # Add vertical line for binary threshold at 10
        plt.axvline(x=9.5, color="red", linestyle="--", linewidth=2)
        plt.text(
            9.4,
            plt.ylim()[1] * 0.95,
            "Pass/Fail Threshold (10)",
            color="red",
            fontsize=12,
            ha="right",
        )

        # Calculate and add percentage labels for fail and pass categories
        fail_count = (G3 < 10).sum()
        fail_pct = (G3 < 10).mean() * 100
        pass_count = (G3 >= 10).sum()
        pass_pct = (G3 >= 10).mean() * 100

        # Add text boxes with statistics for each category
        plt.text(
            5,
            plt.ylim()[1] * 0.85,
            f"Failing: {fail_count} students ({fail_pct:.1f}%)",
            color="red",
            fontsize=12,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.text(
            15,
            plt.ylim()[1] * 0.85,
            f"Passing: {pass_count} students ({pass_pct:.1f}%)",
            color="green",
            fontsize=12,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Add class imbalance information
        imbalance_ratio = pass_count / fail_count
        plt.text(
            0.5,
            0.05,
            f"Class Imbalance Ratio (Pass:Fail): {imbalance_ratio:.2f}:1",
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="center",
            bbox=dict(facecolor="lightyellow", alpha=0.7),
        )

        # Add labels and title
        plt.xlabel("Final Grade (G3)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(
            "Distribution of Final Grades with Binary Classification Threshold",
            fontsize=15,
        )
        plt.grid(alpha=0.3)

        # Set x-ticks to be exactly at grade values
        plt.xticks(range(0, 21))

        # Save the figure
        plt.tight_layout()
        plt.savefig(
            "DataVisualizations/grade_distribution_histogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def feature_correlation_matrix(self):
        """
        Creates a correlation matrix heatmap showing relationships between features
        and target variables.
        """
        # Combine features and targets for correlation analysis
        all_data = pd.concat([self.X, self.raw_targets], axis=1)

        # Convert categorical features to numeric
        # For binary features
        binary_mapping = {
            "yes": 1,
            "no": 0,
            "M": 1,
            "F": 0,
            "GP": 1,
            "MS": 0,
            "U": 1,
            "R": 0,
            "GT3": 1,
            "LE3": 0,
            "T": 1,
            "A": 0,
        }

        # Create a copy to avoid modifying original data
        correlation_data = all_data.copy()

        # Apply binary mapping
        for col in correlation_data.columns:
            if correlation_data[col].dtype == "object":
                # Try binary mapping first
                mapped = correlation_data[col].map(binary_mapping)

                # If binary mapping doesn't work for all values, use one-hot encoding
                if mapped.isna().any():
                    # For non-binary categorical features, use first value as reference (0)
                    unique_values = correlation_data[col].unique()

                    # Create a simple ordinal mapping
                    mapping = {val: idx for idx, val in enumerate(unique_values)}
                    correlation_data[col] = correlation_data[col].map(mapping)
                else:
                    correlation_data[col] = mapped

        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()

        # Create mask for upper triangle to make the plot cleaner
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create the plot
        plt.figure(figsize=(16, 14))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            annot=False,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

        plt.title("Feature Correlation Matrix", fontsize=16)

        # Highlight correlations with target variables
        for i, target in enumerate(["G1", "G2", "G3"]):
            # Get top 5 correlations with this target
            top_corrs = (
                corr_matrix[target].drop(["G1", "G2", "G3"]).abs().nlargest(5).index
            )

            # Annotate these correlations on the heatmap
            for feature in top_corrs:
                try:
                    j = list(corr_matrix.columns).index(feature)
                    k = list(corr_matrix.columns).index(target)

                    if j < k:  # Only annotate in the lower triangle
                        val = corr_matrix.loc[target, feature]
                        plt.text(
                            j + 0.5,
                            k + 0.5,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            bbox=dict(facecolor="white", alpha=0.7),
                        )
                except:
                    continue

        # Save the figure
        plt.tight_layout()
        plt.savefig(
            "DataVisualizations/feature_correlation_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create a second, focused correlation plot just for the grades
        plt.figure(figsize=(10, 8))
        grade_corr = correlation_data[["G1", "G2", "G3"]].corr()
        sns.heatmap(
            grade_corr,
            annot=True,
            cmap="YlGnBu",
            fmt=".3f",
            linewidths=0.5,
            square=True,
            annot_kws={"size": 20},
        )
        plt.title("Correlation Between Grade Variables", fontsize=15)

        plt.tight_layout()
        plt.savefig(
            "DataVisualizations/grades_correlation_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    visualizer = RawDataVisualizer()
    visualizer.feature_distribution_visualization()
    visualizer.class_distribution_barcharts()
    visualizer.grade_distribution_histogram()
    visualizer.feature_correlation_matrix()
