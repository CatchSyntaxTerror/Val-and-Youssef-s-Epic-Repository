import numpy as np

from Assignment1.Task1.ModifiedAdalineGD import ModifiedAdalineGD
from ModifiedLogisticRegressionGD import ModifiedLogisticRegressionGD

def run_test(model, name):
    print(f"\n=== Testing {name} ===")

    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]], dtype=float)
    y = np.array([0,1,1,1])

    model.fit(X, y)
    preds = model.predict(X)

    print("Weights shape:", model.w_.shape)
    print("Losses length:", len(model.losses_))
    print("Predictions:", preds)
    print("Targets:   ", y)

run_test(ModifiedAdalineGD(n_iter=50), "ModifiedAdalineGD")
run_test(ModifiedLogisticRegressionGD(n_iter=50), "ModifiedLogisticRegressionGD")
