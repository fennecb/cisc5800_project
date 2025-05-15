import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, recall_score

from Tasks.BinaryClassification import BinaryClassification
from Tasks.MultiClassification import MultiClassification
from Tasks.Regression import Regression
from SavingResults.OutputCapture import setup_experiment_dir


class MultiExperimentHandler:
    """Runs multiple ML experiments on student performance data and compares results."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory to save experiment results
        """
        self.output_dir = output_dir or f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.results = {
            'binary': {},
            'multi': {},
            'regression': {}
        }
    
    def run_binary_experiments(self, algorithms, threshold=10.0, test_size=0.33, 
                          cv=5, n_jobs=-1, random_state=42, optimize_negative_recall=False):
        """
        Run binary classification experiments with multiple algorithms.
        
        Args:
            algorithms: List of algorithm names to test
            threshold: Grade threshold for passing/failing
            test_size: Test set proportion
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs for grid search
            random_state: Random seed
            optimize_negative_recall: Whether to optimize the probability threshold for negative class recall
            
        Returns:
            Dictionary of experiment results
        """
        results = {}
        
        # Initialize threshold_curves dictionary to store curves for all algorithms
        self.threshold_curves = {}
        
        for algorithm in tqdm(algorithms, desc="Binary Classification", unit='models'):
            # Setup experiment directory with threshold info in name if optimizing
            suffix = 'threshold_tuned' if optimize_negative_recall else 'smote'
            exp_dir, plots_dir = setup_experiment_dir('binary', algorithm, suffix)
            
            # Initialize model
            model = BinaryClassification(threshold=threshold)
            model.plots_dir = plots_dir
            
            # Check class distribution
            class_dist = model.check_class_imbalance()
            
            # Split data
            X_train, X_test, y_train, y_test = model.train_test_split(
                test_size=test_size, 
                random_state=random_state,
                stratify=model.y
            )
            
            # Prepare features
            model.prepare_features(X_train)
            
            # Grid search
            grid_search = model.perform_grid_search(
                X_train, y_train, 
                algorithm, 
                cv=cv, 
                n_jobs=n_jobs
            )
            
            # If optimize_negative_recall is True, find the optimal threshold
            optimal_threshold = 0.5  # Default threshold
            if optimize_negative_recall:
                # Access the trained model correctly from grid_search_results
                if hasattr(model, 'grid_search_results') and 'best_model' in model.grid_search_results:
                    best_model = model.grid_search_results['best_model']
                    
                    # Check if the model can output probabilities
                    if hasattr(best_model, "predict_proba"):
                        # Get probability predictions
                        y_proba = best_model.predict_proba(X_test)[:, 1]
                        
                        # Find optimal threshold for negative class recall
                        optimal_threshold, achieved_recall, threshold_results = model.find_threshold_for_negative_class_recall(
                            target_recall=0.85,  # Target 85% recall for negative class
                            X_test=X_test,
                            y_test=y_test
                        )
                        
                        # Store the threshold curve data for visualization
                        # Extract the data generated during the threshold calculation
                        thresholds = np.linspace(0.01, 0.99, 100)
                        neg_recalls = []
                        pos_recalls = []
                        accuracies = []
                        f1_scores = []
                        
                        for thresh in thresholds:
                            y_pred = (y_proba >= thresh).astype(int)
                            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                            
                            # Calculate metrics
                            neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  # Negative class recall
                            pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Positive class recall
                            accuracy = (tp + tn) / (tp + tn + fp + fn)
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            f1 = 2 * (precision * pos_recall) / (precision + pos_recall) if (precision + pos_recall) > 0 else 0
                            
                            neg_recalls.append(neg_recall)
                            pos_recalls.append(pos_recall)
                            accuracies.append(accuracy)
                            f1_scores.append(f1)
                        
                        # Store all curves for later plotting
                        self.threshold_curves[algorithm] = pd.DataFrame({
                            'threshold': thresholds,
                            'neg_recall': neg_recalls,
                            'pos_recall': pos_recalls,
                            'accuracy': accuracies,
                            'f1': f1_scores
                        })
                        
                        # Re-evaluate with the optimal threshold
                        evaluation_results = model.evaluate_model(X_test, y_test, prediction_threshold=optimal_threshold)
                    else:
                        # For models without probabilistic outputs, use default evaluation
                        evaluation_results = model.evaluate_model(X_test, y_test)
                        print(f"Warning: Model {algorithm} does not support probability outputs. Using default threshold.")
                else:
                    # Fallback if model structure is different
                    evaluation_results = model.evaluate_model(X_test, y_test)
                    print(f"Warning: Model structure for {algorithm} is not as expected. Using default evaluation.")
            else:
                # Standard evaluation without threshold optimization
                evaluation_results = model.evaluate_model(X_test, y_test)
            
            # Save metrics we care about
            results[algorithm] = {
                'accuracy': evaluation_results['accuracy'],
                'balanced_accuracy': evaluation_results['balanced_accuracy'],
                'f1_score': evaluation_results['f1_score'],
                'minority_class_precision': evaluation_results['minority_class_precision'],
                'minority_class_recall': evaluation_results['minority_class_recall'],
                'minority_class_f1': evaluation_results['minority_class_f1'],
                'roc_auc': evaluation_results['roc_auc'],
                'average_precision': evaluation_results['average_precision'],
                'best_params': model.grid_search_results['best_params'],
                'class_distribution': {
                    str(k): int(v) for k, v in class_dist.items()
                }
            }
            
            # Add threshold information if optimized
            if optimize_negative_recall:
                results[algorithm]['probability_threshold'] = optimal_threshold
                    
            # Save results to file
            with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                json.dump(results[algorithm], f, indent=4)
                    
        self.results['binary'] = results
        return results
    
    def _find_optimal_threshold_for_negative_recall(self, y_true, y_prob, num_thresholds=100):
        """
        Find the optimal probability threshold to maximize negative class recall.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            num_thresholds: Number of thresholds to evaluate
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.linspace(0.01, 0.99, num_thresholds)
        best_threshold = 0.5
        best_recall = 0.0
        
        # Find the negative class label (typically 0)
        neg_class = 0
        
        metrics = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate negative class recall (specificity)
            neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate positive class recall (sensitivity)
            pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate overall accuracy and F1 score
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * pos_recall) / (precision + pos_recall) if (precision + pos_recall) > 0 else 0
            
            metrics.append({
                'threshold': threshold,
                'neg_recall': neg_recall,
                'pos_recall': pos_recall,
                'accuracy': accuracy,
                'f1': f1
            })
            
            # Update best threshold if we find better negative class recall
            if neg_recall > best_recall:
                best_recall = neg_recall
                best_threshold = threshold
        
        # Store metrics for later visualization
        self.threshold_metrics = pd.DataFrame(metrics)
        
        print(f"Optimal threshold for negative class recall: {best_threshold:.4f} (Negative recall: {best_recall:.4f})")
        return best_threshold
    
    def _calculate_metrics_with_custom_threshold(self, y_true, y_pred, y_prob):
        """
        Calculate evaluation metrics with a custom threshold.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (using custom threshold)
            y_prob: Predicted probabilities for positive class
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                                   precision_score, recall_score, confusion_matrix,
                                   roc_auc_score, average_precision_score)
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Class-specific metrics
        # Assuming class 1 is the positive/majority class and 0 is negative/minority
        minority_class = 0
        
        # Check if class 0 is actually the minority
        if np.sum(y_true == 0) > np.sum(y_true == 1):
            minority_class = 1
        
        if minority_class == 0:
            # Negative class (0) is minority
            minority_precision = precision_score(y_true, y_pred, pos_label=0)
            minority_recall = recall_score(y_true, y_pred, pos_label=0)
            minority_f1 = f1_score(y_true, y_pred, pos_label=0)
        else:
            # Positive class (1) is minority
            minority_precision = precision_score(y_true, y_pred)
            minority_recall = recall_score(y_true, y_pred)
            minority_f1 = f1_score(y_true, y_pred)
        
        # ROC AUC and Average Precision
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
        except:
            roc_auc = 0.5
            avg_precision = np.sum(y_true == 1) / len(y_true)
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1,
            'minority_class_precision': minority_precision,
            'minority_class_recall': minority_recall,
            'minority_class_f1': minority_f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        }
    
    def _plot_roc_curve_with_threshold(self, y_true, y_prob, threshold, plots_dir, algorithm):
        """
        Plot ROC curve with the selected threshold point highlighted.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold: Selected probability threshold
            plots_dir: Directory to save the plot
            algorithm: Name of the algorithm
        """
        from sklearn.metrics import roc_curve, roc_auc_score
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate AUC
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # Find the closest threshold value in thresholds array
        threshold_idx = np.argmin(np.abs(thresholds - threshold))
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        
        # Mark the selected threshold point
        plt.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red', s=100, 
                   label=f'Threshold = {threshold:.2f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve with Optimal Threshold ({algorithm})', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f'roc_curve_threshold.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve_with_threshold(self, y_true, y_prob, threshold, plots_dir, algorithm):
        """
        Plot Precision-Recall curve with the selected threshold point highlighted.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold: Selected probability threshold
            plots_dir: Directory to save the plot
            algorithm: Name of the algorithm
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate average precision
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Find the closest threshold value in thresholds array
        # Add a 0 at the end of thresholds list as precision_recall_curve returns one fewer threshold
        thresholds = np.append(thresholds, 0)
        threshold_idx = np.argmin(np.abs(thresholds - threshold))
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        
        # Mark the selected threshold point
        plt.scatter(recall[threshold_idx], precision[threshold_idx], color='red', s=100, 
                   label=f'Threshold = {threshold:.2f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve with Optimal Threshold ({algorithm})', fontsize=14)
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f'precision_recall_curve_threshold.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred, plots_dir, algorithm):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            plots_dir: Directory to save the plot
            algorithm: Name of the algorithm
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        plt.xlabel('Predicted labels', fontsize=12)
        plt.ylabel('True labels', fontsize=12)
        plt.title(f'Confusion Matrix ({algorithm})', fontsize=14)
        
        # Add class labels
        tick_marks = np.arange(len(np.unique(y_true))) + 0.5
        plt.xticks(tick_marks, ['Fail', 'Pass'])
        plt.yticks(tick_marks, ['Fail', 'Pass'])
        
        # Display percentages
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'\n{cm_norm[i, j]:.1%}', 
                        ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_threshold.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_metrics(self, plots_dir, algorithm):
        """
        Plot various metrics against different probability thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            plots_dir: Directory to save plots
            algorithm: Name of the algorithm
        """
        if not hasattr(self, 'threshold_metrics'):
            return
        
        df = self.threshold_metrics
        
        plt.figure(figsize=(10, 8))
        plt.plot(df['threshold'], df['neg_recall'], label='Negative Class Recall', linewidth=2)
        plt.plot(df['threshold'], df['pos_recall'], label='Positive Class Recall', linewidth=2)
        plt.plot(df['threshold'], df['accuracy'], label='Accuracy', linewidth=2)
        plt.plot(df['threshold'], df['f1'], label='F1 Score', linewidth=2)
        
        # Find the optimal threshold based on negative recall
        optimal_idx = df['neg_recall'].idxmax()
        optimal_threshold = df.loc[optimal_idx, 'threshold']
        plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.2f}')
        
        plt.xlabel('Probability Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Metrics vs. Threshold ({algorithm})', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(plots_dir, 'threshold_metrics.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_negative_recall_vs_threshold(self, save=True):
        """
        Generate a line plot showing negative class recall vs threshold for each algorithm.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        if not self.results['binary']:
            print("No binary classification results to plot")
            return None
        
        # Check if we have any threshold curves
        if not hasattr(self, 'threshold_curves') or not self.threshold_curves:
            print("No threshold curves available. Run with optimize_negative_recall=True first.")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # For each binary classification model
        for algorithm, curve_data in self.threshold_curves.items():
            plt.plot(curve_data['threshold'], curve_data['neg_recall'], 
                    linewidth=2, label=f'{algorithm}')
                
        plt.xlabel('Probability Threshold', fontsize=12)
        plt.ylabel('Negative Class Recall', fontsize=12)
        plt.title('Negative Class Recall vs Threshold by Algorithm', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        
        if save:
            filepath = os.path.join(self.output_dir, 'negative_recall_vs_threshold.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved negative recall vs threshold chart to {filepath}")
            
        return plt.gcf()
    
    def run_multi_experiments(self, algorithms, n_bins=5, strategy='quantile', 
                             test_size=0.33, cv=5, n_jobs=-1, random_state=42):
        """
        Run multi-class classification experiments with multiple algorithms.
        
        Args:
            algorithms: List of algorithm names to test
            n_bins: Number of grade bins
            strategy: Binning strategy ('quantile', 'uniform', 'grades')
            test_size: Test set proportion
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs for grid search
            random_state: Random seed
            
        Returns:
            Dictionary of experiment results
        """
        results = {}
        
        for algorithm in tqdm(algorithms, desc="Multi-class Classification"):
            # Setup experiment directory
            exp_dir, plots_dir = setup_experiment_dir('multi', algorithm, 'smote')
            
            # Initialize model
            model = MultiClassification(n_bins=n_bins, strategy=strategy)
            model.plots_dir = plots_dir
            
            # Check class distribution
            class_dist = model.check_class_imbalance()
            
            # Split data
            X_train, X_test, y_train, y_test = model.train_test_split(
                test_size=test_size, 
                random_state=random_state,
                stratify=model.y
            )
            
            # Prepare features
            model.prepare_features(X_train)
            
            # Grid search
            grid_search = model.perform_grid_search(
                X_train, y_train, 
                algorithm, 
                cv=cv, 
                n_jobs=n_jobs
            )
            
            # Evaluate model
            eval_results = model.evaluate_model(X_test, y_test)
            
            # Save metrics we care about
            results[algorithm] = {
                'accuracy': eval_results['accuracy'],
                'f1_macro': eval_results['f1_macro'],
                'f1_weighted': eval_results['f1_weighted'],
                'precision_macro': eval_results['precision_macro'],
                'recall_macro': eval_results['recall_macro'],
                'per_class_f1': eval_results['per_class_f1'],
                'best_params': model.grid_search_results['best_params'],
                'class_distribution': {
                    str(k): int(v) for k, v in class_dist.items()
                }
            }
            
            # Save results to file
            with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                json.dump(results[algorithm], f, indent=4)
                
        self.results['multi'] = results
        return results
    
    def run_regression_experiments(self, algorithms, target='G3', 
                                 test_size=0.33, cv=5, n_jobs=-1, random_state=42):
        """
        Run regression experiments with multiple algorithms.
        
        Args:
            algorithms: List of algorithm names to test
            target: Target variable ('G1', 'G2', or 'G3')
            test_size: Test set proportion
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs for grid search
            random_state: Random seed
            
        Returns:
            Dictionary of experiment results
        """
        results = {}
        
        for algorithm in tqdm(algorithms, desc="Regression"):
            # Setup experiment directory
            exp_dir, plots_dir = setup_experiment_dir('regression', algorithm, 'none')
            
            # Initialize model
            model = Regression(target=target)
            model.plots_dir = plots_dir
            
            # Split data
            X_train, X_test, y_train, y_test = model.train_test_split(
                test_size=test_size, 
                random_state=random_state
            )
            
            # Prepare features
            model.prepare_features(X_train)
            
            # Grid search
            grid_search = model.perform_grid_search(
                X_train, y_train, 
                algorithm, 
                cv=cv, 
                n_jobs=n_jobs
            )
            
            # Evaluate model
            eval_results = model.evaluate_model(X_test, y_test)
            
            # Save metrics we care about
            results[algorithm] = {
                'r2': eval_results['r2'],
                'rmse': eval_results['rmse'],
                'mae': eval_results['mae'],
                'mape': eval_results['mape'],
                'explained_variance': eval_results['explained_variance'],
                'best_params': model.grid_search_results['best_params']
            }
            
            # Save results to file
            with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                json.dump(results[algorithm], f, indent=4)
                
        self.results['regression'] = results
        return results
    
    def plot_binary_comparison(self, metrics=None, save=True):
        """
        Generate comparative charts for binary classification experiments.
        
        Args:
            metrics: List of metrics to plot (default: main classification metrics)
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        if not self.results['binary']:
            print("No binary classification results to plot")
            return None
            
        metrics = metrics or ['balanced_accuracy', 'minority_class_recall', 
                             'minority_class_precision', 'minority_class_f1', 'roc_auc']
        results = self.results['binary']
        algorithms = list(results.keys())
        
        # Prepare data for plotting
        plot_data = []
        for alg in algorithms:
            for metric in metrics:
                if metric in results[alg]:
                    plot_data.append({
                        'Algorithm': alg,
                        'Metric': metric,
                        'Value': results[alg][metric]
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Plotting
        plt.figure(figsize=(15, 10))
        
        # For more consistent and visually appealing charts
        colors = sns.color_palette("viridis", len(algorithms))
        g = sns.barplot(x='Metric', y='Value', hue='Algorithm', data=df, palette=colors)
        
        # Add threshold information to title if available
        has_threshold = any('probability_threshold' in results[alg] for alg in algorithms)
        title_suffix = " (with Optimized Negative Recall Threshold)" if has_threshold else ""
        plt.title(f'Binary Classification Performance Comparison{title_suffix}', fontsize=16)
        
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'binary_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved binary comparison chart to {filepath}")
            
        return plt.gcf()
    
    def plot_multi_comparison(self, metrics=None, save=True):
        """
        Generate comparative charts for multi-class classification experiments.
        
        Args:
            metrics: List of metrics to plot (default: main classification metrics)
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        if not self.results['multi']:
            print("No multi-class classification results to plot")
            return None
            
        metrics = metrics or ['accuracy', 'f1_macro', 'f1_weighted', 
                             'precision_macro', 'recall_macro']
        results = self.results['multi']
        algorithms = list(results.keys())
        
        # Prepare data for plotting
        plot_data = []
        for alg in algorithms:
            for metric in metrics:
                if metric in results[alg]:
                    plot_data.append({
                        'Algorithm': alg,
                        'Metric': metric,
                        'Value': results[alg][metric]
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Plotting
        plt.figure(figsize=(15, 10))
        
        colors = sns.color_palette("viridis", len(algorithms))
        g = sns.barplot(x='Metric', y='Value', hue='Algorithm', data=df, palette=colors)
        
        plt.title('Multi-class Classification Performance Comparison', fontsize=16)
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'multi_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved multi-class comparison chart to {filepath}")
            
        return plt.gcf()
    
    def plot_regression_comparison(self, metrics=None, save=True):
        """
        Generate comparative charts for regression experiments.
        
        Args:
            metrics: List of metrics to plot (default: main regression metrics)
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        if not self.results['regression']:
            print("No regression results to plot")
            return None
            
        metrics = metrics or ['r2', 'explained_variance']
        error_metrics = ['rmse', 'mae']  # These are better when lower
        results = self.results['regression']
        algorithms = list(results.keys())
        
        # Split into two plots - one for metrics where higher is better,
        # and one for error metrics where lower is better
        
        # Accuracy metrics
        plot_data = []
        for alg in algorithms:
            for metric in metrics:
                if metric in results[alg]:
                    plot_data.append({
                        'Algorithm': alg,
                        'Metric': metric,
                        'Value': results[alg][metric]
                    })
        
        df = pd.DataFrame(plot_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Accuracy metrics (higher is better)
        if not df.empty:
            colors = sns.color_palette("viridis", len(algorithms))
            g = sns.barplot(x='Metric', y='Value', hue='Algorithm', data=df, 
                           palette=colors, ax=axes[0])
            
            axes[0].set_title('Regression Accuracy Metrics (Higher is Better)', fontsize=16)
            axes[0].set_xlabel('Metric', fontsize=14)
            axes[0].set_ylabel('Score', fontsize=14)
            axes[0].grid(axis='y', alpha=0.3)
            axes[0].set_ylim(0, 1.0)
        
        # Error metrics (lower is better)
        error_data = []
        for alg in algorithms:
            for metric in error_metrics:
                if metric in results[alg]:
                    error_data.append({
                        'Algorithm': alg,
                        'Metric': metric,
                        'Value': results[alg][metric]
                    })
        
        error_df = pd.DataFrame(error_data)
        
        if not error_df.empty:
            g = sns.barplot(x='Metric', y='Value', hue='Algorithm', data=error_df, 
                           palette=colors, ax=axes[1])
            
            axes[1].set_title('Regression Error Metrics (Lower is Better)', fontsize=16)
            axes[1].set_xlabel('Metric', fontsize=14)
            axes[1].set_ylabel('Error', fontsize=14)
            axes[1].grid(axis='y', alpha=0.3)
        
        # Only include one legend for the whole figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Algorithm', bbox_to_anchor=(1.05, 0.5), loc='center left')
        axes[0].get_legend().remove()
        if not error_df.empty:
            axes[1].get_legend().remove()
            
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'regression_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved regression comparison chart to {filepath}")
            
        return fig
    
    def plot_binary_minority_class_metrics(self, save=True):
        """
        Generate a focused chart on minority class performance for binary classification.
        This is important for class imbalance situations.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        if not self.results['binary']:
            print("No binary classification results to plot")
            return None
            
        metrics = ['minority_class_precision', 'minority_class_recall', 'minority_class_f1']
        results = self.results['binary']
        algorithms = list(results.keys())
        
        # Prepare data for plotting
        plot_data = []
        for alg in algorithms:
            for metric in metrics:
                if metric in results[alg]:
                    # Clean up metric name for display
                    display_name = metric.replace('minority_class_', '')
                    plot_data.append({
                        'Algorithm': alg,
                        'Metric': display_name.capitalize(),
                        'Value': results[alg][metric]
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        colors = sns.color_palette("colorblind", len(algorithms))
        g = sns.barplot(x='Algorithm', y='Value', hue='Metric', data=df, palette=colors)
        
        # Add threshold information to title if available
        has_threshold = any('probability_threshold' in results[alg] for alg in algorithms)
        title_suffix = " (with Optimized Negative Recall Threshold)" if has_threshold else ""
        plt.title(f'Performance Metrics for Failing Students (Minority Class){title_suffix}', fontsize=16)
        
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'binary_minority_class_metrics.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved minority class metrics chart to {filepath}")
            
        return plt.gcf()
    
    def plot_threshold_comparison(self, save=True):
        """
        Generate a chart showing the optimal thresholds for each algorithm.
        This is only applicable when optimize_negative_recall is True.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            The figure
        """
        if not self.results['binary']:
            print("No binary classification results to plot")
            return None
            
        # Check if we have threshold data
        results = self.results['binary']
        has_threshold = any('probability_threshold' in results[alg] for alg in results.keys())
        
        if not has_threshold:
            print("No threshold data available. Run with optimize_negative_recall=True first.")
            return None
        
        # Prepare data for plotting
        algorithms = []
        thresholds = []
        neg_recalls = []
        
        for alg, metrics in results.items():
            if 'probability_threshold' in metrics:
                algorithms.append(alg)
                thresholds.append(metrics['probability_threshold'])
                neg_recalls.append(metrics['minority_class_recall'])
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        # Create bar chart with thresholds
        bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(algorithms)))
        
        bar_plot = plt.bar(algorithms, thresholds, color=bar_colors, alpha=0.7)
        
        # Add a line with negative class recall values
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(algorithms, neg_recalls, 'ro-', linewidth=2, label='Negative Class Recall')
        
        # Add threshold values on bars
        for i, v in enumerate(thresholds):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
        
        plt.title('Optimal Probability Thresholds for Negative Class Recall', fontsize=16)
        plt.xlabel('Algorithm', fontsize=14)
        ax1.set_ylabel('Threshold Value', fontsize=14)
        ax2.set_ylabel('Negative Class Recall', color='r', fontsize=14)
        plt.xticks(rotation=45)
        ax1.set_ylim(0, 1.0)
        ax2.set_ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add a legend for the recall line
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'threshold_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved threshold comparison chart to {filepath}")
            
        return plt.gcf()
    
    def save_all_results(self):
        """Save all results to a single JSON file"""
        result_path = os.path.join(self.output_dir, 'all_results.json')
        
        # Convert any non-serializable objects to strings/simple types
        serializable_results = {}
        for model_type, models in self.results.items():
            serializable_results[model_type] = {}
            for alg, metrics in models.items():
                serializable_results[model_type][alg] = {}
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries (like best_params)
                        serializable_results[model_type][alg][metric] = {
                            k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                            for k, v in value.items()
                        }
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        serializable_results[model_type][alg][metric] = value
                    else:
                        serializable_results[model_type][alg][metric] = str(value)
        
        with open(result_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        print(f"Saved all results to {result_path}")


def main():
    parser = argparse.ArgumentParser(description='Run multiple student performance prediction experiments')
    
    parser.add_argument('--model_types', 
                       nargs='+',
                       choices=['binary', 'multi', 'regression'],
                       default=['binary'],
                       help='Types of models to train (binary, multi, or regression)')
    
    parser.add_argument('--algorithms', 
                        nargs='+',
                        default=['random_forest', 'logistic_regression', 'gradient_boosting', 'knn', 'xgboost', 'svm'],
                        help='Algorithms to test')
    
    parser.add_argument('--threshold', 
                       type=float, 
                       default=10.0,
                       help='Threshold for binary classification (default: 10.0)')
    
    parser.add_argument('--n_bins', 
                       type=int, 
                       default=5,
                       help='Number of bins for multi-class classification (default: 5)')
    
    parser.add_argument('--strategy', 
                       choices=['quantile', 'uniform', 'grades'],
                       default='quantile',
                       help='Strategy for multi-class binning (default: quantile)')
    
    parser.add_argument('--optimize_negative_recall',
                       action='store_true',
                       help='Optimize probability threshold for negative class recall')
                       
    parser.add_argument('--target', 
                       choices=['G1', 'G2', 'G3'],
                       default='G3',
                       help='Target variable for regression (default: G3)')
    
    parser.add_argument('--test_size', 
                       type=float, 
                       default=0.33,
                       help='Test set size (default: 0.33)')
    
    parser.add_argument('--cv', 
                       type=int, 
                       default=5,
                       help='Number of cross-validation folds (default: 5)')
    
    parser.add_argument('--n_jobs', 
                       type=int, 
                       default=-1,
                       help='Number of parallel jobs (default: -1)')
    
    parser.add_argument('--random_state', 
                       type=int, 
                       default=42,
                       help='Random state for reproducibility (default: 42)')
    
    parser.add_argument('--output_dir', 
                        type=str,
                        default=None,
                        help='Directory to save experiment results')
    
    args = parser.parse_args()
    
    # Classification algorithms that are shared across binary and multi
    classification_algorithms = [
        'random_forest', 'logistic_regression', 'svm', 'gradient_boosting', 'knn', 'xgboost'
    ]
    
    # Regression algorithms
    regression_algorithms = [
        'random_forest', 'linear_regression', 'gradient_boosting', 
        'ridge', 'lasso', 'svr'
    ]
    
    # Filter algorithms based on specified ones
    if args.algorithms[0] == 'all':
        binary_algorithms = classification_algorithms
        multi_algorithms = classification_algorithms
        regression_algorithms = regression_algorithms
    else:
        binary_algorithms = [a for a in args.algorithms if a in classification_algorithms]
        multi_algorithms = [a for a in args.algorithms if a in classification_algorithms]
        regression_algorithms = [a for a in args.algorithms if a in regression_algorithms]
    
    # Initialize experiment runner
    runner = MultiExperimentHandler(output_dir=args.output_dir)
    
    # Run experiments for each model type
    if 'binary' in args.model_types:
        print(f"\nRunning binary classification with algorithms: {binary_algorithms}")
        runner.run_binary_experiments(
            algorithms=binary_algorithms,
            threshold=args.threshold,
            test_size=args.test_size,
            cv=args.cv,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            optimize_negative_recall=args.optimize_negative_recall
        )
        runner.plot_binary_comparison()
        runner.plot_binary_minority_class_metrics()
        
        # If threshold optimization was performed, also create a threshold comparison plot
        if args.optimize_negative_recall:
            runner.plot_threshold_comparison()
            runner.plot_negative_recall_vs_threshold()
    
    if 'multi' in args.model_types:
        print(f"\nRunning multi-class classification with algorithms: {multi_algorithms}")
        runner.run_multi_experiments(
            algorithms=multi_algorithms,
            n_bins=args.n_bins,
            strategy=args.strategy,
            test_size=args.test_size,
            cv=args.cv,
            n_jobs=args.n_jobs,
            random_state=args.random_state
        )
        runner.plot_multi_comparison()
    
    if 'regression' in args.model_types:
        print(f"\nRunning regression with algorithms: {regression_algorithms}")
        runner.run_regression_experiments(
            algorithms=regression_algorithms,
            target=args.target,
            test_size=args.test_size,
            cv=args.cv,
            n_jobs=args.n_jobs,
            random_state=args.random_state
        )
        runner.plot_regression_comparison()
    
    # Save all results
    runner.save_all_results()
    
    print("\nExperiments completed! Results and plots saved to:", runner.output_dir)


if __name__ == '__main__':
    main()