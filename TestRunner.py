import argparse
import sys
from Tasks.BinaryClassification import BinaryClassification
from Tasks.MultiClassification import MultiClassification
from Tasks.Regression import Regression

def main():
    parser = argparse.ArgumentParser(description='Student Performance Prediction')
    
    parser.add_argument('--model_type', 
                       choices=['binary', 'multi', 'regression'],
                       required=True,
                       help='Type of model to train (binary, multi, or regression)')
    
    parser.add_argument('--algorithm', 
                        choices=['random_forest', 'logistic_regression', 'svm', 'gradient_boosting',
                                'extra_trees', 'naive_bayes', 'knn', 'ada_boost',
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
                       help='Target precision for automatic threshold selection (e.g., 0.9 for 90%)')

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

    args = parser.parse_args()

    regression_algorithms = ['random_forest', 'linear_regression', 'gradient_boosting', 
                             'ridge', 'lasso', 'elasticnet', 'svr']
    classification_algorithms = ['random_forest', 'logistic_regression', 'svm', 'gradient_boosting',
                                 'extra_trees', 'naive_bayes', 'knn', 'ada_boost']
    
    if args.model_type == 'regression' and args.algorithm not in regression_algorithms:
        print(f"Error: {args.algorithm} is not available for regression.")
        print(f"Available algorithms for regression: {', '.join(regression_algorithms)}")
        sys.exit(1)
    elif args.model_type in ['binary', 'multi'] and args.algorithm not in classification_algorithms:
        print(f"Error: {args.algorithm} is not available for classification.")
        print(f"Available algorithms for classification: {', '.join(classification_algorithms)}")
        sys.exit(1)

    print(f"\nInitializing {args.model_type} model with {args.algorithm}...")
    
    if args.model_type == 'binary':
        model = BinaryClassification(threshold=args.threshold)
        print(f"Binary classification with threshold: {args.threshold}")
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
    
    # Split data
    print(f"\nSplitting data (test_size={args.test_size}, random_state={args.random_state})...")
    stratify = None if args.model_type == 'regression' else model.y  # Fixed stratification
    X_train, X_test, y_train, y_test = model.train_test_split(
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=stratify
    )
    
    # Prepare features
    print("\nPreparing features...")
    model.prepare_features(X_train)
    
    # Perform grid search
    print(f"\nPerforming grid search with {args.algorithm} (cv={args.cv}, n_jobs={args.n_jobs})...")
    grid_search = model.perform_grid_search(
        X_train, y_train, 
        args.algorithm, 
        cv=args.cv, 
        n_jobs=args.n_jobs
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    results = model.evaluate_model(X_test, y_test)
    
    # NEW: Enhanced Binary Classification Analysis
    if args.model_type == 'binary':
        # Find optimal threshold based on target precision
        if args.target_precision is not None:
            print(f"\nFinding threshold for {args.target_precision*100}% precision...")
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
        else:
            use_custom_threshold = False
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    if args.model_type == 'binary':
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
        print(f"MAPE: {results['mape']*100:.2f}%")
        print(f"Explained Variance: {results['explained_variance']:.4f}")
    
    print(f"\nBest Parameters:")
    for param, value in model.grid_search_results['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\nBest Cross-validation Score: {model.grid_search_results['best_scores']}")
    
    # Plot results
    print("\nGenerating visualizations...")
    model.plot_results()
    
    # Additional analysis based on model type
    if args.model_type == 'binary':
        # NEW: Enhanced binary analysis
        print("\nGenerating precision-threshold trade-off plot...")
        insights = model.plot_precision_threshold_trade_off()
        
        print("\nKey insights from threshold analysis:")
        print(f"Maximum precision achievable: {insights['max_precision']:.3f}")
        print(f"Threshold for max precision: {insights['threshold_for_max_precision']:.3f}")
        if insights['high_precision_threshold_90']:
            print(f"Threshold for 90%+ precision: {insights['high_precision_threshold_90']:.3f}")
            print(f"Recall at 90% precision: {insights['recall_at_90_precision']:.3f}")
        
        model.plot_probability_distribution()
        model.plot_threshold_analysis()
        
        if args.target_precision is not None and hasattr(model, 'custom_threshold_results'):
            print("\nAnalyzing students predicted to fail with custom threshold...")
            threshold = model.custom_threshold_results['threshold']
            failing_students, summary = model.analyze_failing_students(X_test, threshold=threshold)
            print(f"Students identified as at-risk: {summary['n_predicted_failing']}")
            print(f"Average fail probability: {summary['mean_fail_probability']:.3f}")
            
            # print("\nTop 5 most important features for failing students:")
            # for feature, stats in summary['important_features'][:5]:
            #     print(f"  {feature}: importance={stats['importance']:.3f}, mean={stats['mean_value']:.3f}")
        
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