import subprocess
import concurrent.futures
import multiprocessing

num_cores = multiprocessing.cpu_count()
print(f"Detected {num_cores} CPU cores")

experiments = [
    "--model_type binary --algorithm random_forest --save_results --ensemble_voting single --imbalance_method class_weight",
    "--model_type binary --algorithm logistic_regression --save_results --ensemble_voting single --imbalance_method undersampling",
    "--model_type binary --algorithm gradient_boosting --save_results --ensemble_voting single",
    "--model_type binary --algorithm gradient_boosting --save_results --ensemble_voting single --imbalance_method class_weight",
    "--model_type binary --algorithm knn --save_results --ensemble_voting single --imbalance_method undersampling",
    "--model_type binary --algorithm naive_bayes --save_results --ensemble_voting single",
    "--model_type binary --algorithm naive_bayes --save_results --ensemble_voting single --imbalance_method class_weight",
    "--model_type binary --algorithm xgboost --save_results --ensemble_voting single --imbalance_method class_weight",
    "--model_type regression --algorithm random_forest --save_results --ensemble_voting single",
    "--model_type regression --algorithm svr --save_results --ensemble_voting single",
]

def run_experiment(args):
    """Run a single experiment with the given arguments"""
    cmd = f"python3 TestRunner.py {args} --n_jobs 1"
    print(f"Starting: {cmd}")
    process = subprocess.run(cmd, shell=True)
    return process.returncode == 0

if __name__ == '__main__':
    # Run experiments in parallel, using all available cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(run_experiment, experiments))

    # Check results
    successful = results.count(True)
    failed = results.count(False)
    print(f"Completed {successful} experiments successfully completed, {failed} failed")