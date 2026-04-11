import os

"""
Functions for recording results
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_baseline(config, results, use_test):
    """
    save baseline results
    """
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

def save_dropout(config, results):
    """
    save dropout results
    """
    path = os.path.join(BASE_DIR, "outputs", "tables")
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "dropout_results.log")
    mode = "Bagging" if config["num_models"] > 1 else "Single"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a") as f:
        if not file_exists:
            f.write("Dropout Results:\n")

        f.write(f"{mode} RUN:\n")
        f.write(f"config: {config}\n")
        f.write(f"results: {results}\n")
        f.flush()

def save_others(config, results):
    """
    save dropout results
    """
    path = os.path.join(BASE_DIR, "outputs", "tables")
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "LRGD_results.log")
    file_exists = os.path.exists(file_path)

    with open(file_path, "a") as f:
        if not file_exists:
            f.write("Dropout Results:\n")

        f.write(f"Logistic Regresion:\n")
        f.write(f"config: {config}\n")
        f.write(f"results: {results}\n")
        f.flush()