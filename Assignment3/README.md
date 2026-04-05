# Assignment 3 - Sentiment Classification

Authors: Youssef Amin, Valerie Barker

## Overview
This project builds a neural network to classify IMDb movie reviews.

## Files

- `main.py`  
  Entry point. Loads data, builds the model, and runs basic tests.

- `src/data_loader.py`  
  Loads, preprocesses, shuffles, splits, and vectorizes the dataset.

- `src/model.py`  
  Defines the neural network (fully connected with configurable layers and dropout).

- `aclImdb/`  
  Dataset containing training and test reviews.

## How to Run

```bash
python3 main.py