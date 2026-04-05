import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Load and vectorize data using approach from textbook
"""

def load_raw_reviews(path):
    """
    Read in all reviews from IMDb datasets
    """
    labels = {"pos": 1, "neg": 0}
    data = []

    for split in ("test", "train"):
        for label in ("pos", "neg"):
            p = os.path.join(path, split, label)

            for filename in sorted(os.listdir(p)):
                file_path = os.path.join(p, filename)
                with open(file_path, "r", encoding="utf-8") as inf:
                    review_text = inf.read()
                data.append([review_text, labels[label]])
    
    df = pd.DataFrame(data, columns = ["review", "sentiment"])
    return df

def preprocessor(text):
    """
    clean text: remove none words and make lower case
    """
    text = re.sub(r"<[^>]*>", "", text)
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    return text

def shuffle_data(df, seed=0):
    """
    Shuffle the data.
    """
    np.random.seed(seed)
    return df.reindex(np.random.permutation(df.index)).reset_index(drop=True)

def split_data(df):
    """
    Split data: first 25,000 rows for training, remaining 25,000 for testing.
    """
    X_train = df.loc[:24999, "review"].values
    y_train = df.loc[:24999, "sentiment"].values
    X_test = df.loc[25000:, "review"].values
    y_test = df.loc[25000:, "sentiment"].values

    return X_train, y_train, X_test, y_test

def vectorize_data(X_train, X_test):
    """
    Fit TF-IDF on training data and transform both train and test data.
    """
    vectorizer = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec

def load_data(path="../aclImdb"):
    """
    load, clean, shuffle, split, vectorize
    """
    df = load_raw_reviews(path)
    df["review"] = df["review"].apply(preprocessor)
    df = shuffle_data(df)

    X_train, y_train, X_test, y_test = split_data(df)
    X_train_vec, X_test_vec = vectorize_data(X_train, X_test)

    return X_train_vec, y_train, X_test_vec, y_test