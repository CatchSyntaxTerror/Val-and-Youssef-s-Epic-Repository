import time
"""
A Val classic
"""

class Timer():
    def __init__(self):
        self.time = time.time()

    def stop(self):
        return self.time - time.time()