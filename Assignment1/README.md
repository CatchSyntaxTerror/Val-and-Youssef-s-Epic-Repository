# Assignment 1 — Adaline & Logistic Regression (CS 429/529)

**Team Members:**  
Youssef Amin  
Valerie Barker  

## Overview
This project implements and analyzes Adaline and logistic regression for supervised classification. The bias term is absorbed into the weight vector by appending a constant feature to the input data. Model performance is evaluated using loss convergence on the Iris and Wine datasets.

The project also includes:
- A one-vs-rest perceptron approach for multiclass classification on the full Iris dataset
- Implementations of stochastic gradient descent (SGD) and mini-batch SGD for logistic regression
- Visual comparisons of loss convergence behavior

## Files
- `ModifiedAdalineGD.py` – Adaline with bias absorbed into weights  
- `ModifiedLogisticRegressionGD.py` – Logistic regression with bias absorbed into weights  
- `logisticRegressionSGD.py` – Logistic regression using SGD  
- `logisticMiniBatchSGD.py` – Logistic regression using mini-batch SGD  
- `cerberus.py` – One-vs-rest perceptron implementation for multiclass Iris classification  
- `datasetImport.py` – Dataset loading and preprocessing  
- `compare_losses.py` – Script to train models and generate plots  
- `images/` – Output figures used in the report  

## Requirements
- Python 3
- NumPy
- Pandas
- Matplotlib

## Running the Code
To generate all plots used in the report, run:
```bash
python compare_losses.py
