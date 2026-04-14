import time
import torch
"""
General utility functions
"""

class Timer():
    """
    lol
    """
    def __init__(self):
        self.start = time.time()

    def stop(self):
        return time.time() - self.start

def to_tensors(data):
    """
    convert dict of numpy arrays to PyTorch tensors
    """
    data["xtr"] = torch.tensor(data["xtr"], dtype=torch.float32)
    data["ytr"] = torch.tensor(data["ytr"], dtype=torch.float32)
    data["xtst"] = torch.tensor(data["xtst"], dtype=torch.float32)
    data["ytst"] = torch.tensor(data["ytst"], dtype=torch.float32)
    return data