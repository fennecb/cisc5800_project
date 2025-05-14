from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_recall_curve, average_precision_score,
    f1_score, balanced_accuracy_score, precision_score, recall_score,
    make_scorer, roc_curve
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

from BaseStudentPerformance import BaseStudentPerformance
from Enums.ParamGrids import ParamGrids

import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class BinaryClassification(BaseStudentPerformance):
    def __init__(self, threshold=10, imbalance_method='smote', 
                class_weight_ratio=5.49, ensemble_models=None,
                ensemble_voting='soft', ensemble_threshold=0.4):
        super().__init__()
        self.transform_target_binary(threshold)
        self.threshold = threshold
        self.imbalance_method = imbalance_method
        self.class_weight_ratio = class_weight_ratio
        self.ensemble_models = ensemble_models or ['random_forest', 'logistic_regression', 'xgboost']
        self.ensemble_voting = ensemble_voting
        self.ensemble_threshold = ensemble_threshold
    
    def create_pipeline(self, classifier_name):
        """Create pipeline with chosen imbalance correction method"""
        classifier = self.get_classifier(classifier_name)
        
        # Apply custom class weights if supported
        if hasattr(classifier, 'class_weight') and self.imbalance_method == 'class_weight':
            weight_dict = {0: self.class_weight_ratio, 1: 1.0}
            classifier.set_params(class_weight=weight_dict)
        elif hasattr(classifier, 'class_weight'):
            classifier.set_params(class_weight='balanced')
        
        if self.imbalance_method == 'smote':
            pipeline = ImbPipeline([
                ('binary_encode', self.binary_transformer),
                ('column_transform', self.column_transformer),
                ('smote', SMOTE(random_state=42)),
                ('classifier', classifier)
            ])
        elif self.imbalance_method == 'class_weight':
            pipeline = ImbPipeline([
                ('binary_encode', self.binary_transformer),
                ('column_transform', self.column_transformer),
                ('classifier', classifier)
            ])
        elif self.imbalance_method == 'undersampling':
            pipeline = ImbPipeline([
                ('binary_encode', self.binary_transformer),
                ('column_transform', self.column_transformer),
                ('undersampler', RandomUnderSampler(random_state=42)),
                ('classifier', classifier)
            ])
        else:
            # Default to just preprocessing for ensemble (handled separately)
            pipeline = ImbPipeline([
                ('binary_encode', self.binary_transformer),
                ('column_transform', self.column_transformer),
                ('classifier', classifier)
            ])
        
        return pipeline
    
    def get_classifier(self, classifier_name):
        """Get classifier instance based on name, including XGBoost"""
        from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        
        classifiers = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, tol=1e-3),
            'svm': SVC(random_state=42, probability=True),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'extra_trees': ExtraTreesClassifier(random_state=42),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(),
            'ada_boost': AdaBoostClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        return classifiers.get(classifier_name.lower())
    
    def create_ensemble_pipeline(self):
        """Create an ensemble of models with custom weighting for imbalance correction"""
        estimators = []
        
        for i, model_name in enumerate(self.ensemble_models):
            model = self.get_classifier(model_name)
            
            # Apply custom class weighting to each model that supports it
            if hasattr(model, 'class_weight'):
                weight_dict = {0: self.class_weight_ratio, 1: 1.0}
                model.set_params(class_weight=weight_dict)
            
            # Name each estimator uniquely
            estimators.append((f"{model_name}_{i}", model))
        
        # Create a voting classifier
        voting = self.ensemble_voting
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
        
        # Create the full pipeline
        pipeline = ImbPipeline([
            ('binary_encode', self.binary_transformer),
            ('column_transform', self.column_transformer),
            ('ensemble', ensemble)
        ])
        
        return pipeline
    
    def perform_grid_search(self, X_train, y_train, classifier_name, param_grid=None, cv=5, n_jobs=-1, optimize_for='recall_failing'):
        """Perform grid search optimized for identifying failing students"""
        if self.imbalance_method == 'ensemble':
            # For ensemble, train each model separately with optimal params
            self.fitted_pipelines = {}
            best_individual_scores = {}
            
            for model_name in self.ensemble_models:
                print(f"\nTraining {model_name} for ensemble...")
                
                # Get param grid for this model
                if param_grid is None:
                    model_param_grid_enum = getattr(ParamGrids, f"{model_name.upper()}_BINARY", None)
                    print(getattr(ParamGrids, f"{model_name.upper()}_BINARY", None))
                    print(model_param_grid_enum.value)
                    if model_param_grid_enum is None:
                        model_param_grid_enum = getattr(ParamGrids, model_name.upper(), None)
                    if model_param_grid_enum is None:
                        print(f"No parameter grid found for {model_name}, using default parameters")
                        model_param_grid = {}
                    else:
                        model_param_grid = model_param_grid_enum.value
                        if self.imbalance_method != 'smote':
                            # Remove any parameters related to SMOTE
                            model_param_grid = {k: v for k, v in model_param_grid.items() 
                                                if not k.startswith('smote__')}
                        
                        # Add custom class weight options to parameter grid
                        if 'classifier__class_weight' in model_param_grid:
                            # Define various class weight ratios to try
                            class_weight_options = [
                                {0: 5, 1: 1},
                                # {0: 10, 1: 1},
                                # {0: 20, 1: 1},
                                # {0: 50, 1: 1},
                                'balanced'
                            ]
                            model_param_grid['classifier__class_weight'] = class_weight_options
                
                # Create and optimize individual pipeline
                pipeline = self.create_pipeline(model_name)
                
                # Define scoring metrics focused on minority class recall
                failing_recall_scorer = make_scorer(recall_score, pos_label=0, zero_division=0)
                
                scoring = {
                    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
                    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
                    'f1': make_scorer(f1_score, average='weighted', zero_division=0),
                    'roc_auc': 'roc_auc',
                    'recall_failing': failing_recall_scorer,
                    'precision_failing': make_scorer(precision_score, pos_label=0, zero_division=0),
                    'f1_failing': make_scorer(f1_score, pos_label=0, zero_division=0),
                }
                
                refit_metric = optimize_for if optimize_for in scoring else 'recall_failing'
                print(f'Param grid for {model_name.upper()}')
                print(model_param_grid)
                grid_search = GridSearchCV(
                    pipeline,
                    model_param_grid,
                    cv=cv,
                    scoring=scoring,
                    refit=refit_metric,
                    n_jobs=n_jobs,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store the best model
                self.fitted_pipelines[model_name] = grid_search.best_estimator_
                best_individual_scores[model_name] = {
                    'params': grid_search.best_params_,
                    'recall_failing': grid_search.cv_results_[f'mean_test_recall_failing'][grid_search.best_index_],
                    'precision_failing': grid_search.cv_results_[f'mean_test_precision_failing'][grid_search.best_index_],
                    'f1_failing': grid_search.cv_results_[f'mean_test_f1_failing'][grid_search.best_index_],
                }
                
                print(f"{model_name} best score ({refit_metric}): {best_individual_scores[model_name][refit_metric]:.3f}")
            
            # Now create the ensemble with the best models
            estimators = []
            for model_name, pipeline in self.fitted_pipelines.items():
                # Extract just the classifier part
                classifier = pipeline.named_steps['classifier']
                estimators.append((model_name, classifier))
            
            # Create the voting classifier
            voting = self.ensemble_voting
            ensemble = VotingClassifier(estimators=estimators, voting=voting)
            
            # Create a final pipeline with preprocessing
            self.ensemble_pipeline = ImbPipeline([
                ('binary_encode', self.binary_transformer),
                ('column_transform', self.column_transformer),
                ('ensemble', ensemble)
            ])
            
            # Fit the ensemble pipeline
            self.ensemble_pipeline.fit(X_train, y_train)
            
            # Store as the main fitted pipeline
            self.fitted_pipeline = self.ensemble_pipeline
            
            # Create a summary of the ensemble
            self.grid_search_results = {
                'classifier': 'ensemble',
                'best_params': {model: info['params'] for model, info in best_individual_scores.items()},
                'best_scores': {
                    'individual_models': best_individual_scores
                },
                'best_model': self.ensemble_pipeline,
                'optimized_for': refit_metric
            }
            
            print(f"Ensemble model created with {len(estimators)} models")
            print(f"Voting method: {self.ensemble_voting}")
            print(f"Prediction threshold: {self.ensemble_threshold}" if self.ensemble_voting == 'soft' else "")
            
            return self.ensemble_pipeline
        else:
            # Regular grid search for wanted algorithm
            if param_grid is None:
                param_grid_enum = getattr(ParamGrids, f"{classifier_name.upper()}_BINARY", None)
                if param_grid_enum is None:
                    param_grid_enum = getattr(ParamGrids, classifier_name.upper(), None)
                if param_grid_enum is None:
                    raise ValueError(f"No parameter grid found for {classifier_name}")
                param_grid = param_grid_enum.value

                if self.imbalance_method != 'smote':
                    param_grid = {k: v for k, v in param_grid.items() 
                                                if not k.startswith('smote__')}
                
                # Add custom class weights to parameter grid for class_weight method
                if self.imbalance_method == 'class_weight' and 'classifier__class_weight' in param_grid:
                    # Define various class weight ratios to try
                    class_weight_options = [
                        {0: 5, 1: 1},
                        # {0: 10, 1: 1},
                        # {0: 20, 1: 1},
                        # {0: 50, 1: 1},
                        'balanced'
                    ]
                    param_grid['classifier__class_weight'] = class_weight_options

            pipeline = self.create_pipeline(classifier_name)

            failing_recall_scorer = make_scorer(recall_score, pos_label=0, zero_division=0)

            scoring = {
                'precision': make_scorer(precision_score, average='weighted', zero_division=0),
                'recall': make_scorer(recall_score, average='weighted', zero_division=0),
                'f1': make_scorer(f1_score, average='weighted', zero_division=0),
                'roc_auc': 'roc_auc',
                'recall_failing': failing_recall_scorer,
                'precision_failing': make_scorer(precision_score, pos_label=0, zero_division=0),
                'f1_failing': make_scorer(f1_score, pos_label=0, zero_division=0),
            }

            refit_metric = optimize_for if optimize_for in scoring else 'recall_failing'
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring=scoring,
                refit=refit_metric,
                n_jobs=n_jobs,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.grid_search_results = {
                'classifier': classifier_name,
                'best_params': grid_search.best_params_,
                'best_scores': {metric: grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_] 
                            for metric in scoring.keys()},
                'cv_results': grid_search.cv_results_,
                'best_model': grid_search.best_estimator_,
                'optimized_for': refit_metric
            }

            self.fitted_pipeline = grid_search.best_estimator_

            print(f"Model optimized for: {refit_metric}")
            print(f"Recall for failing students: {self.grid_search_results['best_scores']['recall_failing']:.3f}")
            
            return grid_search
    
    def evaluate_model(self, X_test, y_test, grid_search=None, prediction_threshold=0.5):
        """Evaluate the model with metrics suitable for imbalanced data"""
        # Use the best model from grid search
        if grid_search is None:
            model = self.grid_search_results['best_model']
        else:
            model = grid_search.best_estimator_

        y_pred_proba = model.predict_proba(X_test)
        
        # Apply custom threshold if different from default
        if prediction_threshold != 0.5:
            y_pred = (y_pred_proba[:, 1] >= prediction_threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
            'average_precision': average_precision_score(y_test, y_pred_proba[:, 1]),
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'y_test': y_test,
            'threshold_used': prediction_threshold
        }
        
        # For minority class
        minority_class = y_test.value_counts().idxmin()
        metrics['minority_class_precision'] = precision_score(y_test, y_pred, pos_label=minority_class)
        metrics['minority_class_recall'] = recall_score(y_test, y_pred, pos_label=minority_class)
        metrics['minority_class_f1'] = f1_score(y_test, y_pred, pos_label=minority_class)
        
        self.evaluation_results = metrics
        return metrics
    
    # Preserve all existing methods
    def plot_results(self, X_test=None):
        """
        Plotting for imbalanced binary classification with separate high-resolution files.
        Each plot is saved individually.
        """
        # 1. Confusion Matrix with percentages
        fig_cm = plt.figure(figsize=(10, 8))
        cm = self.evaluation_results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
        plt.title('Normalized Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        self.save_plot(fig_cm, 'confusion_matrix.png')
        plt.close(fig_cm)
        
        # Add special plot for ensemble model that shows per-model contributions
        if self.imbalance_method == 'ensemble' and hasattr(self, 'fitted_pipelines'):
            # Plot performance comparison of individual models vs ensemble
            fig_models = plt.figure(figsize=(12, 8))
            model_names = list(self.fitted_pipelines.keys()) + ['Ensemble']
            
            # Get predictions from individual models
            y_test = self.evaluation_results['y_test']
            minority_recalls = []
            minority_precisions = []
            minority_f1s = []
            
            for model_name, pipeline in self.fitted_pipelines.items():
                model_preds = pipeline.predict(X_test)
                minority_recall = recall_score(y_test, model_preds, pos_label=0)
                minority_precision = precision_score(y_test, model_preds, pos_label=0, zero_division=0)
                minority_f1 = f1_score(y_test, model_preds, pos_label=0, zero_division=0)
                
                minority_recalls.append(minority_recall)
                minority_precisions.append(minority_precision)
                minority_f1s.append(minority_f1)
            
            # Add ensemble scores
            minority_recalls.append(self.evaluation_results['minority_class_recall'])
            minority_precisions.append(self.evaluation_results['minority_class_precision'])
            minority_f1s.append(self.evaluation_results['minority_class_f1'])
            
            # Create bar chart
            x = np.arange(len(model_names))
            width = 0.25
            
            plt.bar(x - width, minority_recalls, width, label='Recall', color='#4ecdc4')
            plt.bar(x, minority_precisions, width, label='Precision', color='#ff6b6b')
            plt.bar(x + width, minority_f1s, width, label='F1', color='#ffb400')
            
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Minority Class (Failing Students) Performance Comparison', fontsize=14)
            plt.xticks(x, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Add value labels above bars
            for i, v in enumerate(minority_recalls):
                plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(minority_precisions):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(minority_f1s):
                plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            
            self.save_plot(fig_models, 'ensemble_model_comparison.png')
            plt.close(fig_models)
        
        # Continue with original plot methods (2-7)
        # 2a. Precision-Recall Curve (Positive Class)
        fig_pr = plt.figure(figsize=(10, 8))
        y_test = self.evaluation_results['y_test']
        y_pred_proba = self.evaluation_results['prediction_probabilities']
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        plt.plot(recall, precision, marker='.', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve (Passing Students)', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        self.save_plot(fig_pr, 'precision_recall_curve.png')
        plt.close(fig_pr)

        # 2a. Precision-Recall Curve (Negative Class)
        fig_pr = plt.figure(figsize=(10, 8))
        y_test = self.evaluation_results['y_test']
        y_pred_proba = self.evaluation_results['prediction_probabilities']
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 0], pos_label=0)
        plt.plot(recall, precision, marker='.', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve (Failing Students)', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        self.save_plot(fig_pr, 'precision_recall_curve_negative.png')
        plt.close(fig_pr)
        
        # 3. ROC Curve
        fig_roc = plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, marker='.', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve (AUC: {self.evaluation_results["roc_auc"]:.3f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save_plot(fig_roc, 'roc_curve.png')
        plt.close(fig_roc)
        
        # 4. Class Distribution
        fig_dist = plt.figure(figsize=(10, 8))
        class_counts = pd.Series(y_test).value_counts()
        plt.bar(class_counts.index, class_counts.values, color=['#ff6b6b', '#4ecdc4'])
        plt.xlabel('Class (0=Fail, 1=Pass)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Class Distribution in Test Set', fontsize=14)
        
        # Add count labels on top of bars
        for i, v in enumerate(class_counts.values):
            plt.text(class_counts.index[i], v + 0.5, str(v), 
                    ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        self.save_plot(fig_dist, 'class_distribution.png')
        plt.close(fig_dist)
        
        # 5. Metrics Summary as a well-formatted table
        fig_metrics = plt.figure(figsize=(12, 10))
        plt.axis('off')
        
        metrics_text = f"""
        Performance Metrics:
        
        Overall:
        • Accuracy: {self.evaluation_results['accuracy']:.3f}
        • Balanced Accuracy: {self.evaluation_results['balanced_accuracy']:.3f}
        • F1 Score: {self.evaluation_results['f1_score']:.3f}
        • Precision: {self.evaluation_results['precision']:.3f}
        • Recall: {self.evaluation_results['recall']:.3f}
        • ROC AUC: {self.evaluation_results['roc_auc']:.3f}
        • Avg Precision: {self.evaluation_results['average_precision']:.3f}
        
        Minority Class (Failing Students):
        • Precision: {self.evaluation_results['minority_class_precision']:.3f}
        • Recall: {self.evaluation_results['minority_class_recall']:.3f}
        • F1: {self.evaluation_results['minority_class_f1']:.3f}
        """
        plt.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center')
        plt.tight_layout()
        self.save_plot(fig_metrics, 'performance_metrics.png')
        plt.close(fig_metrics)
        
        # 6. Feature Importance with real names - skip if using ensemble
        if self.imbalance_method != 'ensemble' and hasattr(self.grid_search_results['best_model'].named_steps['classifier'], 'feature_importances_'):
            fig_imp = plt.figure(figsize=(14, 10))
            best_model = self.grid_search_results['best_model']
            importances = best_model.named_steps['classifier'].feature_importances_
            
            # Get top 15 features for a more comprehensive view
            indices = np.argsort(importances)[::-1][:15]
            
            # Get real feature names
            feature_names = self.get_feature_names_after_preprocessing()
            feature_names_top = [feature_names[i] for i in indices]
            
            # Plot horizontal bar chart for better readability of feature names
            plt.barh(range(len(indices)), importances[indices], color='#4ecdc4')
            plt.yticks(range(len(indices)), feature_names_top)
            plt.xlabel('Importance', fontsize=12)
            plt.title('Top 15 Feature Importances', fontsize=14)
            
            # Add importance values on bars
            for i, v in enumerate(importances[indices]):
                plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
                
            plt.tight_layout()
            self.save_plot(fig_imp, 'feature_importances.png')
            plt.close(fig_imp)
        
        # 7. Combined Precision-Recall-F1 plot showing the balance between these metrics
        fig_combined = plt.figure(figsize=(12, 8))
        
        # Create combined metrics at different thresholds
        thresholds = np.arange(0.05, 0.95, 0.05)
        precision_values = []
        recall_values = []
        f1_values = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)
            
            # Focus on failing students (class 0)
            try:
                precision = precision_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                recall = recall_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                f1 = f1_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                
                precision_values.append(precision)
                recall_values.append(recall)
                f1_values.append(f1)
            except:
                precision_values.append(0)
                recall_values.append(0)
                f1_values.append(0)
        
        plt.plot(thresholds, precision_values, 'b-o', label='Precision', linewidth=2)
        plt.plot(thresholds, recall_values, 'g-s', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_values, 'r-^', label='F1 Score', linewidth=2)
        
        # Mark default threshold and custom threshold for ensemble
        if self.imbalance_method == 'ensemble' and self.ensemble_voting == 'soft':
            plt.axvline(x=0.5, color='black', linestyle='--', label='Default (0.5)')
            plt.axvline(x=self.ensemble_threshold, color='purple', linestyle='-', 
                    label=f'Ensemble threshold ({self.ensemble_threshold:.2f})')
        else:
            plt.axvline(x=0.5, color='black', linestyle='--', label='Default (0.5)')
        
        # Find best F1 threshold
        best_f1_idx = np.argmax(f1_values)
        best_f1_threshold = thresholds[best_f1_idx]
        plt.axvline(x=best_f1_threshold, color='purple', linestyle=':', 
                label=f'Best F1 ({best_f1_threshold:.2f})')
        
        plt.xlabel('Classification Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision-Recall-F1 Trade-off (for Failing Students)', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save_plot(fig_combined, 'precision_recall_f1_tradeoff.png')
        plt.close(fig_combined)
        
        print("All plots have been saved as separate high-resolution files.")
    
    def plot_probability_distribution(self, figsize=(12, 6)):
        """Plot predicted probability distributions by actual class"""
        y_test = self.evaluation_results['y_test']
        y_pred_proba = self.evaluation_results['prediction_probabilities']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # For class 0 (fail)
        fail_probs = y_pred_proba[y_test == 0, 1]
        axes[0].hist(fail_probs, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0].axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
        if self.imbalance_method == 'ensemble' and self.ensemble_voting == 'soft':
            axes[0].axvline(x=self.ensemble_threshold, color='blue', linestyle='-.', 
                        label=f'Ensemble Threshold ({self.ensemble_threshold:.2f})')
        axes[0].set_xlabel('Predicted Probability of Passing')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Probability Distribution for Failing Students')
        axes[0].legend()
        
        # For class 1 (pass)
        pass_probs = y_pred_proba[y_test == 1, 1]
        axes[1].hist(pass_probs, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
        if self.imbalance_method == 'ensemble' and self.ensemble_voting == 'soft':
            axes[1].axvline(x=self.ensemble_threshold, color='blue', linestyle='-.', 
                            label=f'Ensemble Threshold ({self.ensemble_threshold:.2f})')
        axes[1].set_xlabel('Predicted Probability of Passing')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Probability Distribution for Passing Students')
        axes[1].legend()
    
        plt.tight_layout()
        self.save_plot(fig, 'probability_distribution.png')
        
   
    def plot_threshold_analysis(self, figsize=(10, 6)):
        """Analyze model performance at different classification thresholds"""
        y_test = self.evaluation_results['y_test']
        y_pred_proba = self.evaluation_results['prediction_probabilities'][:, 1]
        
        thresholds = np.arange(0, 1.05, 0.05)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics for failing students (class 0)
            try:
                # Check if we have predictions for both classes
                n_predicted_fail = np.sum(y_pred_threshold == 0)
                n_predicted_pass = np.sum(y_pred_threshold == 1)
                
                if n_predicted_fail == 0 or n_predicted_pass == 0:
                    # Skip thresholds that result in only one class being predicted
                    precisions.append(0)
                    recalls.append(0)
                    f1_scores.append(0)
                else:
                    # Use pos_label=0 for failing students and zero_division=0 to avoid warnings
                    precision = precision_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                    recall = recall_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                    f1 = f1_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
            except:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
        
        fig = plt.figure(figsize=figsize)
        plt.plot(thresholds, precisions, label='Precision (Failing Students)', marker='o')
        plt.plot(thresholds, recalls, label='Recall (Failing Students)', marker='s')
        plt.plot(thresholds, f1_scores, label='F1 Score (Failing Students)', marker='^')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold')
        
        # If ensemble method, also show the ensemble threshold
        if self.imbalance_method == 'ensemble' and self.ensemble_voting == 'soft':
            plt.axvline(x=self.ensemble_threshold, color='blue', linestyle='-.', 
                        label=f'Ensemble Threshold ({self.ensemble_threshold:.2f})')
        
        # Find optimal threshold for F1 score of failing students
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='red', linestyle=':', label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')
        
        # Add annotation for threshold interpretation
        plt.text(0.02, 0.9, 'Lower threshold → More students predicted to fail\nHigher threshold → Fewer students predicted to fail', 
                fontsize=9, style='italic', alpha=0.7, transform=plt.gca().transAxes)
        
        plt.xlabel('Probability Threshold for Predicting "Pass" (Class 1)')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs Classification Threshold\n(All metrics calculated for identifying failing students)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self.save_plot(fig, 'threshold_analysis.png')
        
        
        return optimal_threshold
   
    def get_misclassified_samples(self, X_test, include_proba=True):
        """Get samples that were misclassified for further analysis"""
        y_test = self.evaluation_results['y_test']
        y_pred = self.evaluation_results['predictions']
        y_pred_proba = self.evaluation_results['prediction_probabilities']
        
        misclassified_mask = y_test != y_pred
        misclassified_df = X_test[misclassified_mask].copy()
        misclassified_df['actual'] = y_test[misclassified_mask]
        misclassified_df['predicted'] = y_pred[misclassified_mask]
        
        if include_proba:
            misclassified_df['prob_pass'] = y_pred_proba[misclassified_mask, 1]
            misclassified_df['prob_fail'] = y_pred_proba[misclassified_mask, 0]
        
        # Sort by probability to see the most confident misclassifications
        if include_proba:
            misclassified_df = misclassified_df.sort_values('prob_pass', ascending=False)
        
        return misclassified_df
   
    def find_threshold_for_precision(self, target_precision, X_test, y_test, min_samples_per_class=10):
        """
        Find the probability threshold that achieves the target precision.
        Fixed to handle edge cases and avoid warnings.
        
        Parameters:
        - target_precision: float, desired precision (e.g., 0.9 for 90% precision)
        - X_test: test features
        - y_test: test labels
        - min_samples_per_class: minimum samples needed per class for valid calculation
        
        Returns:
        - optimal_threshold: float, threshold that achieves closest to target precision
        - actual_precision: float, achieved precision at that threshold
        - threshold_results: dict, detailed results at the optimal threshold
        """
        # Get prediction probabilities
        if hasattr(self, 'evaluation_results'):
            y_pred_proba = self.evaluation_results['prediction_probabilities'][:, 1]
            y_test_stored = self.evaluation_results['y_test']
        else:
            model = self.grid_search_results['best_model']
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_test_stored = y_test
        
        # Test different thresholds
        thresholds = np.arange(0.01, 1.00, 0.01)
        precisions = []
        recalls = []
        f1_scores = []
        valid_thresholds = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            
            # Check if we have enough samples in both classes
            n_predicted_fail = np.sum(y_pred_threshold == 0)
            n_predicted_pass = np.sum(y_pred_threshold == 1)
            
            # Skip if not enough samples in either class
            if n_predicted_fail < min_samples_per_class or n_predicted_pass < min_samples_per_class:
                continue
            
            # Calculate metrics with zero_division parameter to avoid warnings
            precision = precision_score(y_test_stored, y_pred_threshold, pos_label=0, zero_division=0)
            recall = recall_score(y_test_stored, y_pred_threshold, pos_label=0, zero_division=0)
            f1 = f1_score(y_test_stored, y_pred_threshold, pos_label=0, zero_division=0)
            
            valid_thresholds.append(threshold)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Find threshold closest to target precision
        if not precisions:
            raise ValueError("No valid thresholds found. Try adjusting min_samples_per_class.")
        
        precision_diffs = [abs(p - target_precision) for p in precisions]
        best_idx = np.argmin(precision_diffs)
        
        optimal_threshold = valid_thresholds[best_idx]
        actual_precision = precisions[best_idx]
        actual_recall = recalls[best_idx]
        actual_f1 = f1_scores[best_idx]
        
        # Calculate detailed metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        threshold_results = {
            'threshold': optimal_threshold,
            'precision': actual_precision,
            'recall': actual_recall,
            'f1_score': actual_f1,
            'accuracy': accuracy_score(y_test_stored, y_pred_optimal),
            'balanced_accuracy': balanced_accuracy_score(y_test_stored, y_pred_optimal),
            'confusion_matrix': confusion_matrix(y_test_stored, y_pred_optimal),
            'classification_report': classification_report(y_test_stored, y_pred_optimal),
            'predictions': y_pred_optimal,
            'n_predicted_fail': np.sum(y_pred_optimal == 0),
            'n_predicted_pass': np.sum(y_pred_optimal == 1),
            'n_actual_fail': np.sum(y_test_stored == 0),
            'n_actual_pass': np.sum(y_test_stored == 1)
        }
        
        return optimal_threshold, actual_precision, threshold_results

    def plot_precision_threshold_trade_off(self, figsize=(12, 8)):
        """
        Visualize the trade-off between precision and other metrics as threshold changes.
        Shows where to set threshold for high precision (identifying failing students).
        Fixed to handle edge cases and correct labeling issues.
        """
        # Get prediction probabilities
        if hasattr(self, 'evaluation_results'):
            y_pred_proba = self.evaluation_results['prediction_probabilities'][:, 1]
            y_test = self.evaluation_results['y_test']
        else:
            raise ValueError("No evaluation results found. Run evaluate_model first.")
        
        # Calculate metrics across thresholds
        thresholds = np.arange(0.01, 1.00, 0.01)
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        valid_thresholds = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            
            try:
                # For detecting failing students (class 0), we need to measure precision for class 0
                n_predicted_fail = np.sum(y_pred_threshold == 0)
                n_predicted_pass = np.sum(y_pred_threshold == 1)
                
                # Skip thresholds that result in no predictions for either class
                if n_predicted_fail == 0 or n_predicted_pass == 0:
                    continue
                
                # Calculate metrics with zero_division parameter to avoid warnings
                precision = precision_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                recall = recall_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                f1 = f1_score(y_test, y_pred_threshold, pos_label=0, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred_threshold)
                
                valid_thresholds.append(threshold)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                accuracies.append(accuracy)
            except:
                continue
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Main metrics
        ax1.plot(valid_thresholds, precisions, 'b-', label='Precision (Failing Students)', linewidth=2)
        ax1.plot(valid_thresholds, recalls, 'g-', label='Recall (Failing Students)', linewidth=2)
        ax1.plot(valid_thresholds, f1_scores, 'r-', label='F1 Score (Failing Students)', linewidth=2)
        
        # Add horizontal lines for target precisions
        for target_precision in [0.8, 0.85, 0.9, 0.95]:
            ax1.axhline(y=target_precision, color='gray', linestyle='--', alpha=0.5)
            ax1.text(0.02, target_precision + 0.005, f'{target_precision*100:.0f}%', fontsize=9)
        
        # Mark thresholds
        ax1.axvline(x=0.5, color='black', linestyle='--', label='Default (0.5)')
        if self.imbalance_method == 'ensemble' and self.ensemble_voting == 'soft':
            ax1.axvline(x=self.ensemble_threshold, color='blue', linestyle='-.', 
                        label=f'Ensemble ({self.ensemble_threshold:.2f})')
        
        ax1.set_xlabel('Probability Threshold for Predicting "Pass" (Class 1)')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics vs Threshold for Identifying Failing Students\n(Higher threshold = More conservative in predicting pass)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.05)
        
        ax1.text(0.98, 0.02, 'Low threshold → More students predicted to fail\nHigh threshold → Fewer students predicted to fail', 
                ha='right', va='bottom', fontsize=9, style='italic', alpha=0.7)
        
        # Plot 2: Sample sizes
        n_predicted_fail = []
        n_predicted_pass = []
        
        for threshold in valid_thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            n_predicted_fail.append(np.sum(y_pred_threshold == 0))
            n_predicted_pass.append(np.sum(y_pred_threshold == 1))
        
        ax2.plot(valid_thresholds, n_predicted_fail, 'r-', label='Predicted as Failing', linewidth=2)
        ax2.plot(valid_thresholds, n_predicted_pass, 'g-', label='Predicted as Passing', linewidth=2)
        ax2.axhline(y=np.sum(y_test == 0), color='red', linestyle=':', label='Actual Failing')
        ax2.axhline(y=np.sum(y_test == 1), color='green', linestyle=':', label='Actual Passing')
        
        # Mark thresholds on second plot too
        ax2.axvline(x=0.5, color='black', linestyle='--', label='Default (0.5)')
        if self.imbalance_method == 'ensemble' and self.ensemble_voting == 'soft':
            ax2.axvline(x=self.ensemble_threshold, color='blue', linestyle='-.', 
                        label=f'Ensemble ({self.ensemble_threshold:.2f})')
        
        ax2.set_xlabel('Probability Threshold for Predicting "Pass" (Class 1)')
        ax2.set_ylabel('Number of Students')
        ax2.set_title('Predicted Class Sizes vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        self.save_plot(fig, 'precision_threshold_tradeoff.png')
        
        try:
            high_precision_thresholds = [p for p in precisions if p >= 0.9]
            if high_precision_thresholds:
                high_precision_threshold = valid_thresholds[precisions.index(max(high_precision_thresholds))]
                recall_at_90_precision = recalls[precisions.index(max(high_precision_thresholds))]
            else:
                high_precision_threshold = None
                recall_at_90_precision = None
        except:
            high_precision_threshold = None
            recall_at_90_precision = None
        
        insights = {
            'max_precision': max(precisions) if precisions else 0,
            'threshold_for_max_precision': valid_thresholds[np.argmax(precisions)] if precisions else None,
            'high_precision_threshold_90': high_precision_threshold,
            'recall_at_90_precision': recall_at_90_precision
        }
        
        return insights

    def analyze_failing_students(self, X_test, threshold=None, target_precision=0.9):
        """
        Analyze students predicted to be failing using custom threshold.
        
        Parameters:
        - X_test: test features
        - threshold: custom threshold (if None, finds threshold for target_precision)
        - target_precision: target precision for finding threshold
        """
        # Get the right threshold
        if threshold is None:
            threshold, actual_precision, _ = self.find_threshold_for_precision(target_precision, X_test, None)
            print(f"Using threshold {threshold:.3f} for {actual_precision:.1%} precision")
        
        # Get predictions with custom threshold
        model = self.grid_search_results['best_model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_custom = (y_pred_proba >= threshold).astype(int)
        
        # Analyze students predicted as failing (class 0)
        failing_mask = y_pred_custom == 0
        failing_students = X_test[failing_mask].copy()
        failing_probabilities = y_pred_proba[failing_mask]
        
        # Add probability information
        failing_students['fail_probability'] = 1 - failing_probabilities  # Convert to probability of failing
        failing_students['predicted_class'] = 0
        
        # Sort by confidence (probability of failing)
        failing_students = failing_students.sort_values('fail_probability', ascending=False)
        
        # Create summary statistics
        summary = {
            'n_predicted_failing': len(failing_students),
            'mean_fail_probability': failing_students['fail_probability'].mean(),
            'std_fail_probability': failing_students['fail_probability'].std(),
            'threshold_used': threshold
        }
        
        # Get feature importance for context
        # Skip for ensemble methods since they don't have a simple feature_importances_ attribute
        if self.imbalance_method != 'ensemble' and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            feature_names = self.get_feature_names_after_preprocessing()
            feature_importance = model.named_steps['classifier'].feature_importances_
            
            # Find which features are most important for these failing students
            feature_stats = {}
            for i, feature in enumerate(feature_names):
                if feature in failing_students.columns:
                    feature_stats[feature] = {
                        'importance': feature_importance[i],
                        'mean_value': failing_students[feature].mean(),
                        'std_value': failing_students[feature].std()
                    }
            
            summary['important_features'] = sorted(feature_stats.items(), 
                                                key=lambda x: x[1]['importance'], 
                                                reverse=True)[:10]
        elif self.imbalance_method == 'ensemble':
            # For ensemble models, add a note about feature importance
            summary['ensemble_note'] = "Feature importance not available for ensemble models"
        
        return failing_students, summary
    
    def find_threshold_for_negative_class_accuracy(self, target_accuracy, X_test, y_test=None, min_samples=10):
        """
        Find a probability threshold that achieves the target accuracy for the negative class (failing students).
        
        Parameters:
        ----------
        target_accuracy : float
            Desired accuracy for the negative class (e.g., 0.8 for 80% accuracy)
        X_test : DataFrame
            Test features
        y_test : Series, optional
            Test labels, if not already stored in evaluation_results
        min_samples : int
            Minimum number of samples in each predicted class
            
        Returns:
        -------
        float
            Optimal threshold
        float
            Achieved negative class accuracy
        dict
            Detailed results at the optimal threshold
        """
        # Get prediction probabilities
        if hasattr(self, 'evaluation_results') and y_test is None:
            y_pred_proba = self.evaluation_results['prediction_probabilities'][:, 1]
            y_test_stored = self.evaluation_results['y_test']
        else:
            model = self.grid_search_results['best_model']
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_test_stored = y_test
        
        # Test different thresholds
        thresholds = np.arange(0.01, 1.00, 0.01)
        negative_accuracies = []
        valid_thresholds = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            
            # Check if we have enough samples in both classes
            n_predicted_fail = np.sum(y_pred_threshold == 0)
            n_predicted_pass = np.sum(y_pred_threshold == 1)
            
            if n_predicted_fail < min_samples or n_predicted_pass < min_samples:
                continue
            
            # Calculate negative class accuracy (class 0)
            # This is the ratio of correctly predicted failing students to all actual failing students
            neg_class_mask = y_test_stored == 0
            neg_correct = np.sum((y_pred_threshold[neg_class_mask] == 0))
            neg_total = np.sum(neg_class_mask)
            
            if neg_total > 0:
                neg_accuracy = neg_correct / neg_total
                negative_accuracies.append(neg_accuracy)
                valid_thresholds.append(threshold)
        
        if not negative_accuracies:
            raise ValueError("No valid thresholds found. Try adjusting min_samples parameter.")
        
        # Find threshold that exceeds or is closest to target accuracy
        above_target = [acc >= target_accuracy for acc in negative_accuracies]
        
        if any(above_target):
            # Get the lowest threshold that meets or exceeds target
            best_idx = above_target.index(True)
        else:
            # Get the threshold with highest accuracy if none meet target
            best_idx = np.argmax(negative_accuracies)
        
        optimal_threshold = valid_thresholds[best_idx]
        achieved_accuracy = negative_accuracies[best_idx]
        
        # Generate detailed results at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics for both classes
        cm = confusion_matrix(y_test_stored, y_pred_optimal)
        tn, fp, fn, tp = cm.ravel()
        
        negative_class_metrics = {
            'accuracy': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'recall': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0
        }
        
        threshold_results = {
            'threshold': optimal_threshold,
            'negative_class_accuracy': achieved_accuracy,
            'accuracy': accuracy_score(y_test_stored, y_pred_optimal),
            'balanced_accuracy': balanced_accuracy_score(y_test_stored, y_pred_optimal),
            'confusion_matrix': cm,
            'negative_class_metrics': negative_class_metrics,
            'predictions': y_pred_optimal,
            'n_predicted_fail': np.sum(y_pred_optimal == 0),
            'n_predicted_pass': np.sum(y_pred_optimal == 1),
            'n_actual_fail': np.sum(y_test_stored == 0),
            'n_actual_pass': np.sum(y_test_stored == 1)
        }
        
        return optimal_threshold, achieved_accuracy, threshold_results

    def find_threshold_for_negative_class_recall(self, target_recall, X_test, y_test=None, min_samples=10):
        """
        Find a probability threshold that achieves the target recall for the negative class (failing students).
        
        Parameters:
        ----------
        target_recall : float
            Desired recall for the negative class (e.g., 0.8 for 80% recall)
        X_test : DataFrame
            Test features
        y_test : Series, optional
            Test labels, if not already stored in evaluation_results
        min_samples : int
            Minimum number of samples in each predicted class
            
        Returns:
        -------
        float
            Optimal threshold
        float
            Achieved negative class recall
        dict
            Detailed results at the optimal threshold
        """
        # Get prediction probabilities
        if hasattr(self, 'evaluation_results') and y_test is None:
            y_pred_proba = self.evaluation_results['prediction_probabilities'][:, 1]
            y_test_stored = self.evaluation_results['y_test']
        else:
            model = self.grid_search_results['best_model']
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_test_stored = y_test
        
        # Test different thresholds
        thresholds = np.arange(0.01, 1.00, 0.01)
        recalls = []
        valid_thresholds = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            
            # Check if we have enough samples in both classes
            n_predicted_fail = np.sum(y_pred_threshold == 0)
            n_predicted_pass = np.sum(y_pred_threshold == 1)
            
            if n_predicted_fail < min_samples or n_predicted_pass < min_samples:
                continue
            
            # For negative class (0), recall is the proportion of actual class 0 that are correctly predicted
            tn, fp, fn, tp = confusion_matrix(y_test_stored, y_pred_threshold).ravel()
            
            # Negative class recall = true negative / (true negative + false positive)
            neg_class_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            recalls.append(neg_class_recall)
            valid_thresholds.append(threshold)
        
        if not recalls:
            raise ValueError("No valid thresholds found. Try adjusting min_samples parameter.")
        
        # Find threshold that exceeds or is closest to target recall
        above_target = [rec >= target_recall for rec in recalls]
        
        if any(above_target):
            # Get the lowest threshold that meets or exceeds target
            best_idx = above_target.index(True)
        else:
            # Get the threshold with highest recall if none meet target
            best_idx = np.argmax(recalls)
        
        optimal_threshold = valid_thresholds[best_idx]
        achieved_recall = recalls[best_idx]
        
        # Generate detailed results at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate detailed metrics
        cm = confusion_matrix(y_test_stored, y_pred_optimal)
        tn, fp, fn, tp = cm.ravel()
        
        negative_class_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        negative_class_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        negative_class_f1 = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0
        
        threshold_results = {
            'threshold': optimal_threshold,
            'negative_class_recall': negative_class_recall,
            'negative_class_precision': negative_class_precision,
            'negative_class_f1': negative_class_f1,
            'accuracy': accuracy_score(y_test_stored, y_pred_optimal),
            'balanced_accuracy': balanced_accuracy_score(y_test_stored, y_pred_optimal),
            'confusion_matrix': cm,
            'predictions': y_pred_optimal,
            'n_predicted_fail': np.sum(y_pred_optimal == 0),
            'n_predicted_pass': np.sum(y_pred_optimal == 1),
            'n_actual_fail': np.sum(y_test_stored == 0),
            'n_actual_pass': np.sum(y_test_stored == 1)
        }
        
        return optimal_threshold, achieved_recall, threshold_results