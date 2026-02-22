import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import warnings
import make_classification as mc

"""
Task 4: compare sklearn primal vs dual
"""
def time_fit(model, tx, ty):
    """
    time the fit function
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        t0 = time.time()
        model.fit(tx, ty)
        t1 = time.time()
        run_time = t1 - t0
        return run_time

def append_table(table, d, n, time):
    table.append({"d": d, "n": n, "time": time})

def get_losses(dual: bool, iters, tx, ty, d, n):
    if dual: model = LinearSVC(loss="hinge", dual=dual, random_state=0)
    else: model = LinearSVC(loss="squared_hinge", dual=dual, random_state=0)
    losses = []
    for i in iters:
        model.set_params(max_iter=i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(tx, ty)
        fxs = model.decision_function(tx)
        loss = np.maximum(0, 1-ty * fxs)
        losses.append(np.mean(loss))
    return losses

def plot_loss(model_name, losses, iters, d, n):
    plt.figure()
    plt.plot(iters, losses)
    plt.xlabel("Max Iterations")
    plt.ylabel("Average Loss")
    plt.title(f"{model_name} Losses")
    plt.savefig(f"images/{model_name}_d{d}_n{n}.png")
    plt.close()

def printTable(label, rows, filename="outputs/task4_results.csv"):
    "Prints table in a csv"
    with open(filename, "a") as f:
        f.write(f"{label} Results:\n\n")
        for row in rows:
            f.write(f"{row}\n")
        f.write("\n")
    print(f"Results saved to {filename}")


ds = [10, 50, 100]
ns = [500, 5000, 50000]
iters = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
p_rows = []
d_rows = []
for d in ds:
    for n in ns:
        primal_model = LinearSVC(loss="squared_hinge", dual=False)
        dual_model = LinearSVC(loss="hinge", dual=True)
        data = mc.load_data(d, n)
        trainingX = data["tx"]
        trainingy = data["ty"]
        
        # time fit and build table
        p_time = time_fit(primal_model, trainingX, trainingy)
        d_time = time_fit(dual_model, trainingX, trainingy)
        append_table(p_rows, d, n, p_time)
        append_table(d_rows, d, n, d_time)
        
        # plot
        p_losses = get_losses(False, iters, trainingX, trainingy, d, n)
        d_losses = get_losses(True, iters, trainingX, trainingy, d, n)
        plot_loss("Primal", p_losses, iters, d, n)
        plot_loss("Dual", d_losses, iters, d, n)

printTable("Primal", p_rows)
printTable("Dual", d_rows)    