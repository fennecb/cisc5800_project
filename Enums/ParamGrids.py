from enum import Enum


class ParamGrids(Enum):
    # ==================== BINARY CLASSIFICATION GRIDS ====================
    RANDOM_FOREST_BINARY = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__class_weight": ["balanced", "balanced_subsample", None],
        "classifier__min_samples_split": [2, 5, 10],
        "smote__k_neighbors": [3, 5, 7],
    }

    LOGISTIC_REGRESSION_BINARY = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__class_weight": ["balanced"],
        "classifier__solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
        "classifier__max_iter": [2500, 5000, 10000],
        "smote__k_neighbors": [3, 5, 7],
    }

    SVM_BINARY = {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf", "linear", "sigmoid"],
        "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "smote__k_neighbors": [3, 5, 7],
    }

    GRADIENT_BOOSTING_BINARY = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [5, 7],
        "classifier__min_samples_split": [5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "smote__k_neighbors": [3, 5, 7],
    }

    KNN_BINARY = {
        "classifier__n_neighbors": [3, 5, 7, 9, 11],
        "classifier__weights": ["uniform", "distance"],
        "classifier__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "classifier__p": [1, 2],  # 1 for manhattan_distance, 2 for euclidean_distance
        "smote__k_neighbors": [3, 5, 7],
    }

    XGBOOST_BINARY = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_child_weight": [3, 5, 7],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.8, 0.9, 1.0],
        "smote__k_neighbors": [3, 5, 7],
    }

    # ==================== MULTI-CLASS CLASSIFICATION GRIDS ====================
    RANDOM_FOREST_MULTICLASS = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7, None],
        "classifier__class_weight": ["balanced", "balanced_subsample", None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    LOGISTIC_REGRESSION_MULTICLASS = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__class_weight": ["balanced"],
        "classifier__solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
        "classifier__max_iter": [2500, 5000, 10000],
        "classifier__multi_class": ["auto", "ovr", "multinomial"],
    }

    SVM_MULTICLASS = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__kernel": ["rbf", "linear", "poly", "sigmoid"],
        "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "classifier__class_weight": ["balanced", None],
        "classifier__degree": [2, 3, 4],
        "classifier__decision_function_shape": ["ovo", "ovr"],
    }

    GRADIENT_BOOSTING_MULTICLASS = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__subsample": [0.8, 0.9, 1.0],
    }

    KNN_MULTICLASS = {
        "classifier__n_neighbors": [3, 5, 7, 9, 11],
        "classifier__weights": ["uniform", "distance"],
        "classifier__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "classifier__p": [1, 2],
    }

    # ==================== REGRESSION GRIDS ====================
    RANDOM_FOREST_REGRESSION = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [3, 5, 7, None],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", "log2", None],
    }

    LINEAR_REGRESSION_REGRESSION = {
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False],
    }

    RIDGE_REGRESSION = {
        "regressor__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "regressor__fit_intercept": [True, False],
        "regressor__solver": [
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
        ],
    }

    LASSO_REGRESSION = {
        "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "regressor__fit_intercept": [True, False],
        "regressor__max_iter": [1000, 5000, 10000],
        "regressor__selection": ["cyclic", "random"],
    }

    SVR_REGRESSION = {
        "regressor__C": [0.1, 1, 10, 100],
        "regressor__kernel": ["rbf", "linear", "sigmoid"],
        "regressor__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "regressor__epsilon": [0.01, 0.1, 0.2, 0.5],
    }

    GRADIENT_BOOSTING_REGRESSION = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "regressor__max_depth": [3, 5, 7],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__loss": ["squared_error", "absolute_error", "huber", "quantile"],
        "regressor__subsample": [0.8, 0.9, 1.0],
    }
