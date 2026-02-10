# Assignment 1 — Adaline & Logistic Regression (CS 429/529)

This repository contains the implementation and experiments for **Assignment 1**, covering:
- Adaline and Logistic Regression (bias-absorbed versions)
- Loss convergence comparisons on real datasets
- (Upcoming) multiclass classification and SGD variants

The project is organized by **task number** to directly match the assignment rubric.

---

## Project Structure

```
Assignment1/
├── BookCode/
│   ├── BookAdalineGD.py
│   └── BookLogisticRegressionGD.py
│
├── Task1/
│   ├── ModifiedAdalineGD.py
│   ├── ModifiedLogisticRegressionGD.py
│   └── TaskOneTest.py
│
├── Task2/
│   ├── datasetImport.py
│   └── compare_losses.py
│
├── Task3/
│
├── Task4/
│
└── README.md
```

## Task 1 — Bias-Absorbed Models

Modify Adaline and Logistic Regression so the bias term is absorbed into the weight vector by appending a column of ones to the input.

### Files
- `Assignment1/Task1/ModifiedAdalineGD.py`
- `Assignment1/Task1/ModifiedLogisticRegressionGD.py`
- `Assignment1/Task1/TaskOneTest.py`

### Run Task 1 test
From the **repository root**:
```bash
python Assignment1/Task1/TaskOneTest.py
```

---

## Task 2 — Loss Convergence Comparison

Compare the **loss convergence** of Adaline vs Logistic Regression on:
- **Wine dataset** (class 1 vs class 2)
- **Iris dataset** (setosa vs versicolor)

### Files
- `Assignment1/Task2/datasetImport.py`
- `Assignment1/Task2/compare_losses.py`

### Run dataset loader
```bash
python Assignment1/Task2/datasetImport.py
```

### Run loss comparison
```bash
python Assignment1/Task2/compare_losses.py
```


