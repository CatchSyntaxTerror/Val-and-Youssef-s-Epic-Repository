# Linear SVM Implementation and Analysis
Authors: Youssef Amin and Valerie Barker\
This project implements a Linear Support Vector Machine (SVM) using stochastic gradient descent (SGD), evaluates its scalability, and analyzes the primal and dual formulations of sklearn’s LinearSVC.

## Requirements

The project requires:

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn


## Files

- **LinearSVC.py** – Implementation of the linear SVM using stochastic gradient descent.
- **make_classification.py** – Generates linearly separable synthetic datasets with configurable dimension and size.
- **scalability.py** – Runs scalability experiments and records runtime and loss across dataset scales.
- **compare_sklearn.py** – Compares primal and dual formulations of sklearn's `LinearSVC`.
- **test_LinearSVC.py** – Demonstrates and visualizes the behavior of the implemented classifier.
- **MarginBand.py** – Utility for visualizing the decision boundary and margin band in 2D.
- **test_make_classification.py** – Simple visualization script for generated 2D datasets.

## Folders

- **images/** – Contains all generated plots, including loss convergence curves, decision boundaries, and primal vs dual comparisons.
- **datasets/** – Stores saved synthetic datasets used for scalability experiments.
- **outputs/** – Contains CSV result tables for Task 3 (scalability) and Task 4 (primal vs dual comparison).

## Summary

The project demonstrates how linear SVMs behave under different dataset scales and highlights practical differences between primal and dual optimization strategies.