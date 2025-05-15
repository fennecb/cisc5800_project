import argparse
import sys
import os
from Tasks.BinaryClassification import BinaryClassification
from Tasks.MultiClassification import MultiClassification
from Tasks.Regression import Regression
from SavingResults.OutputCapture import OutputCapture, setup_experiment_dir

import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    parser = argparse.ArgumentParser(
        description='Student Performance Prediction: Train and evaluate models on student performance data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help
        epilog='Example: python TestRunner.py --model_type binary --algorithm random_forest --threshold 12'
    )
    
    parser.add_argument('--model_type', 
                       choices=['binary', 'multi', 'regression'],
                       required=True,
                       help='Type of model to train (binary, multi, or regression)')
    
    parser.add_argument('--algorithm', 
                        choices=['random_forest', 'logistic_regression', 'svm', 'gradient_boosting',
                                'extra_trees', 'naive_bayes', 'knn', 'ada_boost', 'xgboost',
                                'linear_regression', 'ridge', 'lasso', 'elasticnet', 'svr',
                                'bayesian_ridge', 'huber'],
                        required=True,
                        help='Algorithm to use')

    parser.add_argument('--threshold', 
                       type=float, 
                       default=10.0,
                       help='Threshold for binary classification (default: 10.0)')

    parser.add_argument('--prediction_threshold', 
                       type=float, 
                       default=0.5,
                       help='Custom threshold for binary predictions (default: 0.5)')

    parser.add_argument('--target_precision', 
                       type=float, 
                       default=None,
                       help='Target precision for automatic threshold selection (e.g., 0.9 for 90%%)')

    parser.add_argument('--detailed_binary_analysis', 
                       action='store_true',
                       help='Perform detailed analysis for binary classification')
    
    parser.add_argument('--n_bins', 
                       type=int, 
                       default=5,
                       help='Number of bins for multi-class classification (default: 5)')
    
    parser.add_argument('--strategy', 
                       choices=['quantile', 'uniform', 'grades'],
                       default='quantile',
                       help='Strategy for multi-class binning (default: quantile)')
    
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

    parser.add_argument('--imbalance_method', 
                       choices=['smote', 'class_weight', 'ensemble', 'undersampling'],
                       default='smote',
                       help='Method for handling class imbalance (default: smote)')
    
    parser.add_argument('--ensemble_models', 
                       type=str,
                       nargs='+',
                       default=['random_forest', 'logistic_regression', 'xgboost'],
                       choices=['random_forest', 'logistic_regression', 'xgboost', 
                               'gradient_boosting', 'svm', 'extra_trees', 'knn'],
                       help='Models to use in ensemble (when ensemble method is selected)')
    
    parser.add_argument('--ensemble_voting', 
                       choices=['hard', 'soft', 'single'],
                       default='single',
                       help='Voting method for ensemble (default: single (no ensemble))')
    
    parser.add_argument('--ensemble_threshold', 
                       type=float, 
                       default=0.4,
                       help='Prediction threshold for ensemble soft voting (default: 0.4)')

    parser.add_argument('--save_results', 
                        action='store_true',
                        help='Save experiment results to disk')
    
    parser.add_argument('--target_negative_accuracy', 
                        type=float, 
                        default=None,
                        help='Target accuracy for negative class (failing students) for threshold selection')

    parser.add_argument('--target_negative_recall', 
                        type=float, 
                        default=None,
                        help='Target recall for negative class (failing students) for threshold selection')

    args = parser.parse_args()

    regression_algorithms = ['random_forest', 'linear_regression', 'gradient_boosting', 
                             'ridge', 'lasso', 'elasticnet', 'svr', 'bayesian_ridge', 'huber']
    classification_algorithms = ['random_forest', 'logistic_regression', 'svm', 'gradient_boosting',
                                 'extra_trees', 'naive_bayes', 'knn', 'ada_boost', 'xgboost']
    
    if args.model_type == 'regression' and args.algorithm not in regression_algorithms:
        print(f"Error: {args.algorithm} is not available for regression.")
        print(f"Available algorithms for regression: {', '.join(regression_algorithms)}")
        sys.exit(1)
    elif args.model_type in ['binary', 'multi'] and args.algorithm not in classification_algorithms:
        print(f"Error: {args.algorithm} is not available for classification.")
        print(f"Available algorithms for classification: {', '.join(classification_algorithms)}")
        sys.exit(1)

    experiment_dir = None
    plots_dir = None
    if args.save_results:
        if args.ensemble_voting:
            experiment_dir, plots_dir = setup_experiment_dir(
                args.model_type, 
                args.algorithm, 
                args.imbalance_method, 
                args.ensemble_voting, 
                args.ensemble_models,
                args.ensemble_threshold
            )
        else:
            experiment_dir, plots_dir = setup_experiment_dir(args.model_type, args.algorithm, args.imbalance_method)
        print(f"Experiment results will be saved to: {experiment_dir}")

    log_file = None
    if args.save_results:
        log_file = os.path.join(experiment_dir, "experiment_log.txt")
        output_capture = OutputCapture(log_file)
    else:
        pass

    with output_capture:
        # Print experiment configuration
        print(f"\n{'='*30} EXPERIMENT CONFIGURATION {'='*30}")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        print(f"{'='*80}\n")
    print(f"\nInitializing {args.model_type} model with {args.algorithm}...")
    
    if args.model_type == 'binary':
        if args.imbalance_method == 'ensemble':
            # For ensemble method, algorithm parameter is ignored
            model = BinaryClassification(
                threshold=args.threshold,
                imbalance_method=args.imbalance_method,
                ensemble_models=args.ensemble_models,
                ensemble_voting=args.ensemble_voting,
                ensemble_threshold=args.ensemble_threshold
            )
            print(f"Binary classification with threshold: {args.threshold}")
            print(f"Imbalance method: {args.imbalance_method}")
            print(f"Ensemble models: {', '.join(args.ensemble_models)}")
            print(f"Ensemble voting: {args.ensemble_voting}")
            if args.ensemble_voting == 'soft':
                print(f"Ensemble threshold: {args.ensemble_threshold}")
        else:
            # For regular methods, use the specified algorithm
            model = BinaryClassification(
                threshold=args.threshold,
                imbalance_method=args.imbalance_method,
            )
            print(f"Binary classification with threshold: {args.threshold}")
            print(f"Imbalance method: {args.imbalance_method}")
        
        if args.save_results:
            model.plots_dir = plots_dir
            # model.generate_single_model_visualizations(output_dir=plots_dir)
        if args.target_precision is not None:
            print(f"Target precision: {args.target_precision}")
        else:
            print(f"Custom prediction threshold: {args.prediction_threshold}")
    elif args.model_type == 'multi':
        model = MultiClassification(n_bins=args.n_bins, strategy=args.strategy)
        print(f"Multi-class classification with {args.n_bins} bins using {args.strategy} strategy")
    else:  # regression
        model = Regression(target=args.target)
        print(f"Regression with target: {args.target}")
    
    # Check class imbalance (for classification)
    if args.model_type != 'regression':
        print("\nClass Distribution:")
        model.check_class_imbalance()

    print(f"\nSplitting data (test_size={args.test_size}, random_state={args.random_state})...")
    stratify = None if args.model_type == 'regression' else model.y
    X_train, X_test, y_train, y_test = model.train_test_split(
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=stratify
    )

    print("\nPreparing features...")
    model.prepare_features(X_train)

    print(f"\nPerforming grid search {'with ensemble models' if args.imbalance_method == 'ensemble' else f'with {args.algorithm}'} (cv={args.cv}, n_jobs={args.n_jobs})...")
    
    # For ensemble method, we don't need to specify the algorithm
    if args.model_type == 'binary' and args.imbalance_method == 'ensemble':
        grid_search = model.perform_grid_search(
            X_train, y_train, 
            None,  # No specific algorithm for ensemble
            cv=args.cv, 
            n_jobs=args.n_jobs
        )
    else:
        grid_search = model.perform_grid_search(
            X_train, y_train, 
            args.algorithm, 
            cv=args.cv, 
            n_jobs=args.n_jobs
        )

    print("\nEvaluating model on test set...")
    if args.ensemble_voting == 'soft':
        print(f"Using custom ensemble threshold: {args.ensemble_threshold}")
        results = model.evaluate_model(X_test, y_test, prediction_threshold=args.ensemble_threshold)
    elif (args.model_type == 'binary' and args.prediction_threshold != 0.5):
        print(f"Using custom prediction threshold: {args.prediction_threshold}")
        results = model.evaluate_model(X_test, y_test, prediction_threshold=args.prediction_threshold)
    else:
        results = model.evaluate_model(X_test, y_test)

    if args.model_type == 'binary':
        # Find optimal threshold based on target precision
        if args.target_precision is not None:
            print(f"\nFinding threshold for {args.target_precision*100}%% precision...")
            try:
                threshold, actual_precision, threshold_results = model.find_threshold_for_precision(
                    args.target_precision, X_test, y_test
                )
                print(f"Optimal threshold: {threshold:.3f}")
                print(f"Achieved precision: {actual_precision:.3f}")
                print(f"Recall at this precision: {threshold_results['recall']:.3f}")
                print(f"F1 score at this precision: {threshold_results['f1_score']:.3f}")
                
                # Store threshold results for later use
                model.custom_threshold_results = threshold_results
                use_custom_threshold = True
                
            except Exception as e:
                print(f"Could not find threshold for target precision: {e}")
                print("Using default threshold...")
                use_custom_threshold = False

        elif args.target_negative_accuracy is not None:
            print(f"\nFinding threshold for {args.target_negative_accuracy*100}% negative class accuracy...")
            threshold, actual_accuracy, threshold_results = model.find_threshold_for_negative_class_accuracy(
                args.target_negative_accuracy, X_test, y_test
            )
            print(f"Optimal threshold: {threshold:.3f}")
            print(f"Achieved negative class accuracy: {actual_accuracy:.3f}")

        elif args.target_negative_recall is not None:
            print(f"\nFinding threshold for {args.target_negative_recall*100}% negative class recall...")
            threshold, actual_recall, threshold_results = model.find_threshold_for_negative_class_recall(
                args.target_negative_recall, X_test, y_test
            )
            print(f"Optimal threshold: {threshold:.3f}")
            print(f"Achieved negative class recall: {actual_recall:.3f}")

        else:
            use_custom_threshold = False
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    if args.model_type == 'binary':
        if args.prediction_threshold != 0.5:
            print(f"\nResults with Custom Threshold ({args.prediction_threshold}):")
        else:
            print("\nResults with Default Threshold (0.5):")
        if hasattr(model, 'custom_threshold_results'):
            # Show results with custom threshold
            custom_results = model.custom_threshold_results
            print("\nWith Custom Threshold (Optimized for Precision):")
            print(f"Threshold: {custom_results['threshold']:.3f}")
            print(f"Accuracy: {custom_results['accuracy']:.4f}")
            print(f"Balanced Accuracy: {custom_results['balanced_accuracy']:.4f}")
            print(f"F1 Score: {custom_results['f1_score']:.4f}")
            print(f"Precision: {custom_results['precision']:.4f}")
            print(f"Recall: {custom_results['recall']:.4f}")
            print(f"Students predicted to fail: {custom_results['n_predicted_fail']}")
            print(f"Students predicted to pass: {custom_results['n_predicted_pass']}")
            print(f"Actual failing students: {custom_results['n_actual_fail']}")
            print(f"Actual passing students: {custom_results['n_actual_pass']}")
            
            print("\nDefault Threshold (0.5) Results:")
        
        if args.imbalance_method == 'ensemble' and args.ensemble_voting == 'soft':
            print(f"\nEnsemble with threshold {args.ensemble_threshold}:")
        
        # Show default results
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"Average Precision: {results['average_precision']:.4f}")
        print(f"\nMinority Class Metrics:")
        print(f"Precision: {results['minority_class_precision']:.4f}")
        print(f"Recall: {results['minority_class_recall']:.4f}")
        print(f"F1: {results['minority_class_f1']:.4f}")
        
    elif args.model_type == 'multi':
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Macro: {results['f1_macro']:.4f}")
        print(f"F1 Weighted: {results['f1_weighted']:.4f}")
        print(f"Precision Macro: {results['precision_macro']:.4f}")
        print(f"Recall Macro: {results['recall_macro']:.4f}")
        print(f"\nPer-class F1 scores:")
        for class_label, f1 in results['per_class_f1'].items():
            print(f"Class {class_label}: {f1:.4f}")
    else:  # regression
        print(f"R² Score: {results['r2']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"MAPE: {results['mape']*100:.2f}%%")
        print(f"Explained Variance: {results['explained_variance']:.4f}")
    
    # Print best parameters
    print(f"\nBest Parameters:")
    if args.imbalance_method == 'ensemble':
        for model_name, params in model.grid_search_results['best_params'].items():
            print(f"\n{model_name.capitalize()}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
    else:
        for param, value in model.grid_search_results['best_params'].items():
            print(f"  {param}: {value}")
    
    # Print best scores
    if args.imbalance_method == 'ensemble':
        print("\nBest Individual Model Scores:")
        for model_name, scores in model.grid_search_results['best_scores'].get('individual_models', {}).items():
            print(f"  {model_name}: recall_failing={scores.get('recall_failing', 'N/A'):.4f}, "
                  f"precision_failing={scores.get('precision_failing', 'N/A'):.4f}")
    else:
        print(f"\nBest Cross-validation Score: {model.grid_search_results['best_scores']}")
    
    # Plot results
    print("\nGenerating visualizations...")
    model.plot_results(X_test)
    
    # Additional analysis based on model type
    if args.model_type == 'binary':
        print("\nGenerating precision-threshold trade-off plot...")
        insights = model.plot_precision_threshold_trade_off()
        
        print("\nKey insights from threshold analysis:")
        print(f"Maximum precision achievable: {insights['max_precision']:.3f}")
        print(f"Threshold for max precision: {insights['threshold_for_max_precision']:.3f}")
        if insights['high_precision_threshold_90']:
            print(f"Threshold for 90%%+ precision: {insights['high_precision_threshold_90']:.3f}")
            print(f"Recall at 90%% precision: {insights['recall_at_90_precision']:.3f}")
        
        model.plot_probability_distribution()
        model.plot_threshold_analysis()
        
        if args.target_precision is not None and hasattr(model, 'custom_threshold_results'):
            print("\nAnalyzing students predicted to fail with custom threshold...")
            threshold = model.custom_threshold_results['threshold']
            failing_students, summary = model.analyze_failing_students(X_test, threshold=threshold)
            print(f"Students identified as at-risk: {summary['n_predicted_failing']}")
            print(f"Average fail probability: {summary['mean_fail_probability']:.3f}")
            
            # If using ensemble, we might not have feature importances
            if 'important_features' in summary:
                print("\nTop 5 most important features for failing students:")
                for feature, stats in summary['important_features'][:5]:
                    print(f"  {feature}: importance={stats['importance']:.3f}, mean={stats['mean_value']:.3f}")
            elif 'ensemble_note' in summary:
                print(f"\nNote: {summary['ensemble_note']}")
        
        misclassified = model.get_misclassified_samples(X_test)
        print(f"\nMisclassified samples: {len(misclassified)}")
        
    elif args.model_type == 'multi':
        model.plot_confusion_matrix_detailed()
        model.plot_per_class_metrics()
        model.plot_prediction_confidence()
        misclassified_patterns = model.analyze_misclassifications(X_test)
    else:  # regression
        model.plot_residual_analysis()
        model.plot_prediction_error_distribution()
        intervals = model.analyze_prediction_intervals(X_test, confidence=0.95)
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()