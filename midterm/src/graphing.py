import os
import numpy as np
import matplotlib.pyplot as plt
"""
I figured we could use this file to add graphing stuff. 
I save images in output/images and tables in output/tables
"""

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_ROOT, "output")

def log_raw_labels(y_train, y_test, filepath):
    """
    makes a table of the number of samples in each label
    it stores the table in output/tables
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def get_stats(y):
        classes, counts = np.unique(y, return_counts=True)
        return list(zip(classes, counts))
    
    training_stats = get_stats(y_train)
    test_stats = get_stats(y_test)

    with open(filepath, "w") as f:
        f.write("Training Data\n")
        f.write("------------------\n")
        f.write("Classes\t|\tCounts\n")
        for cls, cos in training_stats:
            f.write(f"{cls}\t\t|\t{cos}\n")

        f.write("\nTest Data\n")
        f.write("------------------\n")
        f.write("Classes\t|\tCounts\n")
        for cls, cos in test_stats:
            f.write(f"{cls}\t\t|\t{cos}\n")

def plot_num_labels(y, filepath, title):
    """
    plots the nummber of times a label appears in the data
    stored in outputs/images
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    classes, counts = np.unique(y, return_counts=True)

    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Counts")
    plt.title(title)
    plt.savefig(filepath)
    plt.close()

def record_raw_labels(y_train, y_test, mnist:bool):
    """
    Wrapper for plot_raw_data() and log_raw_data()
    """
    if mnist:
        table_path = os.path.join(OUTPUT_DIR, "tables", "label_count_mnist.log")
        train_graph = os.path.join(OUTPUT_DIR, "graphs", "label_count_mnist_train.png")
        test_graph  = os.path.join(OUTPUT_DIR, "graphs", "label_count_mnist_test.png")
        title = "MNIST Data Distribution"
    else:
        table_path = os.path.join(OUTPUT_DIR, "tables", "label_count_fashion.log")
        train_graph = os.path.join(OUTPUT_DIR, "graphs", "label_count_fashion_train.png")
        test_graph  = os.path.join(OUTPUT_DIR, "graphs", "label_count_fashion_test.png")
        title = "Fashion-MNIST Data Distribution"

    log_raw_labels(y_train, y_test, table_path)
    plot_num_labels(y_train, train_graph, title + " (Training)")
    plot_num_labels(y_test, test_graph, title + " (Test)")


def record_test(error_v, error_t, time, C, gamma, degree, ker, comps, pca:bool):
    """
    makes a table of the test results saves to results_tbale.csv
    """
    if pca: table_path = os.path.join(OUTPUT_DIR, "tables", f"pca_{comps}_{ker}.log")
    else: table_path = os.path.join(OUTPUT_DIR, "tables", f"lda_{ker}.log")
    with open(table_path, "a") as f:
        match ker:
            case "linear": f.write(f"C = {C}, error_v: {error_v:.3f}, error_t: {error_t:.3f}, time: {time:.3f}\n")
            case "rbf": f.write(f"C = {C}, gamma = {gamma}, error_v: {error_v:.3f}, error_t: {error_t:.3f}, time: {time:.3f}\n")
            case "poly": f.write(f"C = {C}, gamma = {gamma}, degree = {degree}, error_v: {error_v:.3f}, error_t: {error_t:.3f}, time: {time:.3f}\n")

def record_final(error_test, time, C, gamma, degree, ker, comps, pca:bool):
    """
    record final result for tuning
    """
    if pca: table_path = os.path.join(OUTPUT_DIR, "tables", f"pca_{comps}_{ker}.log")
    else: table_path = os.path.join(OUTPUT_DIR, "tables", f"lda_{ker}.log")
    with open(table_path, "a") as f:
        match ker:
            case "linear": f.write(f"C = {C}, Final Error: {error_test:.3f}, time: {time:.3f}\n") 
            case "rbf": f.write(f"C = {C}, gamma = {gamma}, Final Error: {error_test:.3f}, time: {time:.3f}\n") 
            case "poly":f.write(f"C = {C}, gamma = {gamma}, degree = {degree}, Final Error: {error_test:.3f}, time: {time:.3f}\n")

    