import os
import sys
import datetime
import matplotlib.pyplot as plt
from io import StringIO
from contextlib import redirect_stdout

def setup_experiment_dir(model_type, algorithm, approach="smote", ensemble_voting="single", ensemble_models='na', ensemble_threshold=0.4):
    """Create a directory for saving experiment results"""
    # Create a timestamp for unique experiment identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment name
    if ensemble_voting == 'single':
        experiment_name = f"{timestamp}_{model_type}_{algorithm}_{approach}"
    elif ensemble_voting == 'soft':
        experiment_name = f"{timestamp}_{model_type}_{ensemble_voting}_threshold_{ensemble_threshold}_ensemble_with_{'+'.join(ensemble_models)}"
    else:
        experiment_name = f"{timestamp}_{model_type}_{ensemble_voting}_ensemble_with_{'+'.join(ensemble_models)}"
    
    # Create base directory for experiments if it doesn't exist
    base_dir = "experiment_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create experiment-specific directory
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir)
    
    # Create subdirectories for plots and logs
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir)
    
    return experiment_dir, plots_dir

class OutputCapture:
    """Class to capture and save terminal output while still displaying it"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
    
    def __enter__(self):
        self.file = open(self.file_path, 'w')
        self.tee_stdout = TeeStdout(self.original_stdout, self.file)
        sys.stdout = self.tee_stdout
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.file.close()

class TeeStdout:
    """Class that duplicates output to both terminal and file"""
    def __init__(self, original_stdout, file):
        self.original_stdout = original_stdout
        self.file = file
    
    def write(self, message):
        self.original_stdout.write(message)
        self.file.write(message)
    
    def flush(self):
        self.original_stdout.flush()
        self.file.flush()