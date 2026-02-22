import time
import numpy as np
import matplotlib.pyplot as plt
import make_classification as mc
from LinearSVC import LinearSVC


def train_model(d, n, u, eta, n_iter, C, seed):
    """
    Generate data set, learn the model and compute the run time. 
    returns: rune time, final loss and the model.
    """
    trainingX, trainingy, testX, testy, a = mc.make_classification(
        d=d, n=n, u=u, seed=seed
    )
    mc.save_data(d, n, trainingX, trainingy)
    
    model = LinearSVC(eta=eta, n_iter=n_iter, random_state=seed)
    t0 = time.time()
    model.fit(trainingX, trainingy, C)
    t1 = time.time()
    final_loss = float(model.losses_[-1])
    run_time = t1 - t0
    return run_time, final_loss, model


ds = [10, 50, 100]
ns = [500, 5000, 50000]
seed = 1
u = 100
eta = 0.000001
n_iter = 50
C = 200

rows = []

for d in ds:
    for n in ns:
        # print(f"Running d = {d}, n = {n}")
        run_time, final_loss, model = train_model(
            d=d, n=n, u=u, eta=eta, n_iter=n_iter, C=C, seed=seed
        )
        
        rows.append(
            {
                "d": d,
                "n": n,
                "u": u,
                "eta": eta,
                "n_iter": n_iter,
                "C": C,
                "run_time": run_time,
                "final_loss": final_loss,
            })
        
        plt.figure()
        plt.plot(model.losses_)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Convergence d = {d}, n = {n}")
        plt.savefig(f"images/loss_convergence_d{d}_n{n}.png")
        plt.close()
        
        # print(f"run time: {run_time}, loss: {loss}")


# Prints table in a csv
filename="outputs/task3_results.csv"
with open(filename, "w") as f:
    f.write("Scalability Results:\n\n")
    for row in rows:
        f.write(f"{row}\n")
print(f"Results saved to {filename}")
