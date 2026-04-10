import os

"""
Functions for recording results
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_baseline(config, results, use_test):
    path = os.path.join(BASE_DIR, "outputs", "tables")
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "baseline_results.log")
    mode = "TEST" if use_test else "VALIDATION"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a") as f:
        if not file_exists:
            f.write("Baseline Results:\n")

        f.write(f"{mode} RUN:\n")
        f.write(f"config: {config}\n")
        f.write(f"results: {results}\n")
        f.flush()
