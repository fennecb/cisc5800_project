import subprocess
import concurrent.futures
import multiprocessing

num_cores = multiprocessing.cpu_count()
print(f"Detected {num_cores} CPU cores")

experiments = [
    "--model_type binary --algorithm logistic_regression --ensemble_voting single --save_results --prediction_threshold .45",
    "--model_type binary --algorithm logistic_regression --ensemble_voting single --save_results --prediction_threshold .44",
    "--model_type binary --algorithm logistic_regression --ensemble_voting single --save_results --prediction_threshold .46",
    "--model_type binary --algorithm logistic_regression --ensemble_voting single --save_results --prediction_threshold .47",
    "--model_type binary --algorithm logistic_regression --ensemble_voting single --save_results --prediction_threshold .48",
]


def run_experiment(args):
    """Run a single experiment with the given arguments"""
    cmd = f"python TestRunner.py {args} --n_jobs 1"
    print(f"Starting: {cmd}")
    process = subprocess.run(cmd, shell=True)
    return process.returncode == 0


if __name__ == "__main__":
    # Run experiments in parallel, using all available cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(run_experiment, experiments))

    # Check results
    successful = results.count(True)
    failed = results.count(False)
    print(f"Completed {successful} experiments successfully completed, {failed} failed")
