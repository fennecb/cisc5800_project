from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_recall_curve, average_precision_score,
    f1_score, balanced_accuracy_score, precision_score, recall_score,
    make_scorer, roc_curve
)
from sklearn.model_selection import GridSearchCV

from BaseStudentPerformance import BaseStudentPerformance
from Enums.ParamGrids import ParamGrids

class BinaryClassification(BaseStudentPerformance):
    def __init__(self, threshold=10):
        super().__init__()
        self.transform_target_binary(threshold)
        self.threshold = threshold
    
    def create_pipeline(self, classifier_name):
        """Create pipeline with SMOTE for binary classification"""
        classifier = self.get_classifier(classifier_name)
        
        if hasattr(classifier, 'class_weight'):
            classifier.set_params(class_weight='balanced')
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('binary_encode', self.binary_transformer),
            ('column_transform', self.column_transformer),
            ('smote', SMOTE(random_state=42)),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def perform_grid_search(self, X_train, y_train, classifier_name, param_grid=None, cv=5, n_jobs=-1, optimize_for='recall_failing'):
        """Perform grid search optimized for identifying failing students"""
        # Get the parameter grid
        if param_grid is None:
            param_grid_enum = getattr(ParamGrids, f"{classifier_name.upper()}_BINARY", None)
            if param_grid_enum is None:
                # Fallback to old naming convention
                param_grid_enum = getattr(ParamGrids, classifier_name.upper(), None)
            if param_grid_enum is None:
                raise ValueError(f"No parameter grid found for {classifier_name}")
            param_grid = param_grid_enum.value
        
        # Create pipeline
        pipeline = self.create_pipeline(classifier_name)
        
        # Define custom scorer for recall of failing students (class 0)
        failing_recall_scorer = make_scorer(recall_score, pos_label=0)
        
        # Use multiple scoring metrics
        scoring = {
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'roc_auc': 'roc_auc',
            'recall_failing': failing_recall_scorer,  # Key metric for your use case
            'precision_failing': make_scorer(precision_score, pos_label=0),
            'f1_failing': make_scorer(f1_score, pos_label=0),
        }
        
        # Choose what to optimize for
        refit_metric = optimize_for if optimize_for in scoring else 'recall_failing'
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit=refit_metric,  # Optimize for recall of failing students
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.grid_search_results = {
            'classifier': classifier_name,
            'best_params': grid_search.best_params_,
            'best_scores': {metric: grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_] 
                        for metric in scoring.keys()},
            'cv_results': grid_search.cv_results_,
            'best_model': grid_search.best_estimator_,
            'optimized_for': refit_metric
        }
        
        # Store fitted pipeline for feature names
        self.fitted_pipeline = grid_search.best_estimator_
        
        # Print optimization target
        print(f"Model optimized for: {refit_metric}")
        print(f"Recall for failing students: {self.grid_search_results['best_scores']['recall_failing']:.3f}")
        
        return grid_search
    
    def evaluate_model(self, X_test, y_test, grid_search=None):
        """Evaluate the model with metrics suitable for imbalanced data"""
        # Use the best model from grid search
        if grid_search is None:
            model = self.grid_search_results['best_model']
        else:
            model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
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
            'y_test': y_test  # Store for plotting
        }
        
        # For minority class
        minority_class = y_test.value_counts().idxmin()
        metrics['minority_class_precision'] = precision_score(y_test, y_pred, pos_label=minority_class)
        metrics['minority_class_recall'] = recall_score(y_test, y_pred, pos_label=minority_class)
        metrics['minority_class_f1'] = f1_score(y_test, y_pred, pos_label=minority_class)
        
        self.evaluation_results = metrics
        return metrics
    
    def plot_results(self, figsize=(15, 10)):
        """Enhanced plotting for imbalanced binary classification"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Confusion Matrix with percentages
        cm = self.evaluation_results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Normalized Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Precision-Recall Curve
        y_test = self.evaluation_results['y_test']
        y_pred_proba = self.evaluation_results['prediction_probabilities']
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        axes[0, 1].plot(recall, precision, marker='.')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True)
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        axes[1, 0].plot(fpr, tpr, marker='.')
        axes[1, 0].plot([0, 1], [0, 1], 'k--')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title(f'ROC Curve (AUC: {self.evaluation_results["roc_auc"]:.3f})')
        
        # 4. Class Distribution
        class_counts = pd.Series(y_test).value_counts()
        axes[1, 1].bar(class_counts.index, class_counts.values)
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Class Distribution')
        
        # 5. Metrics Summary
        metrics_text = f"""
        Accuracy: {self.evaluation_results['accuracy']:.3f}
        Balanced Accuracy: {self.evaluation_results['balanced_accuracy']:.3f}
        F1 Score: {self.evaluation_results['f1_score']:.3f}
        Precision: {self.evaluation_results['precision']:.3f}
        Recall: {self.evaluation_results['recall']:.3f}
        ROC AUC: {self.evaluation_results['roc_auc']:.3f}
        Avg Precision: {self.evaluation_results['average_precision']:.3f}
        
        Minority Class Metrics:
        Precision: {self.evaluation_results['minority_class_precision']:.3f}
        Recall: {self.evaluation_results['minority_class_recall']:.3f}
        F1: {self.evaluation_results['minority_class_f1']:.3f}
        """
        axes[2, 0].text(0.1, 0.1, metrics_text, fontsize=12, verticalalignment='top')
        axes[2, 0].axis('off')
        
        # 6. Feature Importance with real names
        best_model = self.grid_search_results['best_model']
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            importances = best_model.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            # Get real feature names
            feature_names = self.get_feature_names_after_preprocessing()
            feature_names_top = [feature_names[i] for i in indices]
            
            axes[2, 1].bar(range(len(indices)), importances[indices])
            axes[2, 1].set_xticks(range(len(indices)))
            axes[2, 1].set_xticklabels(feature_names_top, rotation=45, ha='right')
            axes[2, 1].set_title('Top 10 Feature Importances')
            axes[2, 1].set_ylabel('Importance')
        else:
            axes[2, 1].text(0.5, 0.5, 'Feature importance not available', ha='center', va='center')
            axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_probability_distribution(self, figsize=(12, 6)):
        """Plot predicted probability distributions by actual class"""
        y_test = self.evaluation_results['y_test']
        y_pred_proba = self.evaluation_results['prediction_probabilities']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # For class 0 (fail)
        fail_probs = y_pred_proba[y_test == 0, 1]
        axes[0].hist(fail_probs, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0].axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
        axes[0].set_xlabel('Predicted Probability of Passing')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Probability Distribution for Failing Students')
        axes[0].legend()
        
        # For class 1 (pass)
        pass_probs = y_pred_proba[y_test == 1, 1]
        axes[1].hist(pass_probs, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
        axes[1].set_xlabel('Predicted Probability of Passing')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Probability Distribution for Passing Students')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
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
        
        plt.figure(figsize=figsize)
        plt.plot(thresholds, precisions, label='Precision (Failing Students)', marker='o')
        plt.plot(thresholds, recalls, label='Recall (Failing Students)', marker='s')
        plt.plot(thresholds, f1_scores, label='F1 Score (Failing Students)', marker='^')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold')
        
        # Find optimal threshold for F1 score of failing students
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='red', linestyle=':', label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')
        
        # Add annotation for threshold interpretation
        plt.text(0.02, 0.8, 'Lower threshold → More students predicted to fail\nHigher threshold → Fewer students predicted to fail', 
                fontsize=9, style='italic', alpha=0.7, transform=plt.gca().transAxes)
        
        plt.xlabel('Probability Threshold for Predicting "Pass" (Class 1)')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs Classification Threshold\n(All metrics calculated for identifying failing students)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
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
                # But we need to be careful about undefined metrics
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
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Main metrics
        ax1.plot(valid_thresholds, precisions, 'b-', label='Precision (Failing Students)', linewidth=2)
        ax1.plot(valid_thresholds, recalls, 'g-', label='Recall (Failing Students)', linewidth=2)
        ax1.plot(valid_thresholds, f1_scores, 'r-', label='F1 Score (Failing Students)', linewidth=2)
        
        # Add precision target lines
        for target_precision in [0.8, 0.85, 0.9, 0.95]:
            ax1.axhline(y=target_precision, color='gray', linestyle='--', alpha=0.5)
            ax1.text(0.02, target_precision + 0.005, f'{target_precision*100:.0f}%', fontsize=9)
        
        ax1.set_xlabel('Probability Threshold for Predicting "Pass" (Class 1)')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics vs Threshold for Identifying Failing Students\n(Higher threshold = More conservative in predicting pass)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.05)
        
        # Add annotation about threshold interpretation
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
        
        ax2.set_xlabel('Probability Threshold for Predicting "Pass" (Class 1)')
        ax2.set_ylabel('Number of Students')
        ax2.set_title('Predicted Class Sizes vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Return some key insights with proper handling of edge cases
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
        failing_students['fail_probability'] = failing_probabilities
        failing_students['predicted_class'] = 0
        
        # Sort by confidence (probability of failing)
        failing_students = failing_students.sort_values('fail_probability', ascending=True)
        
        # Create summary statistics
        summary = {
            'n_predicted_failing': len(failing_students),
            'mean_fail_probability': failing_students['fail_probability'].mean(),
            'std_fail_probability': failing_students['fail_probability'].std(),
            'threshold_used': threshold
        }
        
        # Get feature importance for context
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
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
        
        return failing_students, summary
