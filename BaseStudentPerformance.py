import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os

from Enums.ColumnTypes import ColumnTypes

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BaseStudentPerformance:
    """Base class for all student performance prediction tasks"""

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

    def prepare_features(self):
        """Create preprocessing components - shared across all tasks"""
        from sklearn.preprocessing import FunctionTransformer

        def binary_encoder(X):
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

            X_encoded = X.copy()
            for col in X_encoded.columns:
                if X_encoded[col].dtype == "object":
                    mapped = X_encoded[col].map(binary_mapping)
                    X_encoded[col] = mapped.fillna(X_encoded[col])
            return X_encoded

        self.numeric_features = ColumnTypes.NUMERIC_FEATURES.value
        self.binary_features = ColumnTypes.BINARY_FEATURES.value
        self.categorical_features = ColumnTypes.CATEGORICAL_FEATURES_TO_ENCODE.value
        self.ordinal_features = ColumnTypes.ORDINAL_FEATURES.value

        self.binary_transformer = FunctionTransformer(binary_encoder, validate=False)
        self.numeric_transformer = StandardScaler()
        self.categorical_transformer = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
        self.ordinal_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

        self.column_transformer = ColumnTransformer(
            [
                (
                    "num",
                    self.numeric_transformer,
                    self.numeric_features + self.binary_features,
                ),
                ("cat", self.categorical_transformer, self.categorical_features),
                ("ord", self.ordinal_transformer, self.ordinal_features),
            ],
            remainder="drop",
        )

        return None

    def get_feature_names_after_preprocessing(self):
        """Get actual feature names after preprocessing - shared across all tasks"""
        if not hasattr(self, "fitted_pipeline"):
            return None

        feature_names = []
        column_transformer = self.fitted_pipeline.named_steps["column_transform"]

        for name, transformer, features in column_transformer.transformers_:
            if name == "num":
                feature_names.extend(features)
            elif name == "cat":
                if hasattr(transformer, "get_feature_names_out"):
                    cat_names = transformer.get_feature_names_out(features)
                else:
                    cat_names = [
                        f"{feat}_{val}"
                        for feat, vals in zip(features, transformer.categories_)
                        for val in vals[1:]
                    ]
                feature_names.extend(cat_names)
            elif name == "ord":
                feature_names.extend(features)

        return feature_names

    def train_test_split(self, test_size=0.33, random_state=42, stratify=None):
        """Split data - shared across all tasks"""
        kwargs = {"test_size": test_size, "random_state": random_state}

        if stratify is not None:
            kwargs["stratify"] = stratify

        return train_test_split(self.X, self.y, **kwargs)

    def check_class_imbalance(self):
        """Check distribution - shared for classification tasks"""
        class_counts = pd.Series(self.y).value_counts()
        class_proportions = class_counts / len(self.y) * 100

        print("Class Distribution:")
        for label, count in class_counts.items():
            print(f"Class {label}: {count} samples ({class_proportions[label]:.1f}%)")

        return class_counts

    def get_classifier(self, classifier_name):
        """Get classifier instance based on name"""
        from sklearn.neighbors import KNeighborsClassifier

        classifiers = {
            "random_forest": RandomForestClassifier(random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, tol=1e-3),
            "svm": SVC(random_state=42, probability=True),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "knn": KNeighborsClassifier(),
        }
        return classifiers.get(classifier_name.lower())

    def get_regressor(self, regressor_name):
        """Get regressor instance based on name"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor

        regressors = {
            "random_forest": RandomForestRegressor(random_state=42),
            "linear_regression": LinearRegression(),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "ridge": Ridge(),
            "lasso": Lasso(random_state=42),
            "svr": SVR(),
            "knn": KNeighborsRegressor(),
        }
        return regressors.get(regressor_name.lower())

    def create_base_pipeline(self, estimator):
        """Create base pipeline with preprocessing - to be extended by subclasses"""
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps = [
            ("binary_encode", self.binary_transformer),
            ("column_transform", self.column_transformer),
            ("estimator", estimator),
        ]

        return ImbPipeline(steps)

    def plot_metrics_summary(self, metrics, ax=None):
        """Plot metrics summary - shared across tasks"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # This will be different for regression vs classification
        # Make it flexible enough to handle both
        pass

    def save_results(self, results, filename):
        """Save results to file - shared across tasks"""
        import json

        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

    def load_results(self, filename):
        """Load results from file - shared across tasks"""
        import json

        with open(filename, "r") as f:
            return json.load(f)

    def transform_target_binary(self, threshold=10):
        """Convert target to binary classification"""
        final_grades = self.raw_targets["G3"]
        self.y = (final_grades >= threshold).astype(int)
        self.task_type = "binary_classification"
        return self.y

    def transform_target_multiclass(self, n_bins=5, strategy="quantile"):
        """Convert target to multi-class classification"""
        final_grades = self.raw_targets["G3"]

        if strategy == "quantile":
            self.y = pd.qcut(final_grades, q=n_bins, labels=False)
        elif strategy == "uniform":
            self.y = pd.cut(final_grades, bins=n_bins, labels=False)
        else:
            # Custom bins for A, B, C, D, F grading
            bins = [-1, 5, 9, 13, 16, 20]
            labels = ["F", "D", "C", "B", "A"]
            self.y = pd.cut(final_grades, bins=bins, labels=labels)

        self.task_type = "multiclass_classification"
        return self.y

    def transform_target_regression(self, target="G3"):
        """Keep target as continuous for regression"""
        self.y = self.raw_targets[target]
        self.task_type = "regression"
        return self.y

    def save_plot(self, fig=None, filename="plot.png"):
        """Save the current plot to file"""
        if fig is None:
            fig = plt.gcf()  # Get current figure

        # Save figure if plots_dir is defined
        if hasattr(self, "plots_dir") and self.plots_dir is not None:
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {filepath}")
            return filepath
        else:
            print("Warning: plots_dir not set, plot not saved")
            return None
