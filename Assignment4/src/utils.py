import time
import torch

"""
General utility functions
"""


class Timer:
    """lol"""
    def __init__(self):
        self.start = time.time()

    def stop(self):
        return time.time() - self.start


def to_tensors(data):
    """convert dict of numpy arrays to PyTorch tensors"""
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


def to_lists(data):
    """ convert tensors to lists for printing/output"""
    return {k: v.tolist() for k, v in data.items()}


def get_selection(str, items):
    """ prompt the user and return input """
    
    print(str)
    for i, j in enumerate(items, start=1): print(f"{i}) {j}")
    return items[int(input("")) - 1]
