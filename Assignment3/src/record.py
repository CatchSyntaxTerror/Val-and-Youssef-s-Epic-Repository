import os
import csv

"""
Functions for recording results
"""
OUTDIR = os.path.join("data", "tables")
os.makedirs(OUTDIR, exist_ok=True)

def save_baseline(config, results, use_test):
    path = os.path.join("data", "tables")
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "baseline_results.log")
    mode = "TEST" if use_test else "VALIDATION"

    with open(file_path, "a") as f:
        if not os.path.exists(file_path):
            f.write("Baseline Results:\n\n")
            
        f.write(f"{mode} RUN:\n")
        f.write(f"config: {config}\n")
        f.write(f"results: {results}\n")
        f.write("\n")