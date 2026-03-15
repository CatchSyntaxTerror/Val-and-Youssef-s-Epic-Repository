
"""
Run project analyses and generate results.
"""

class Result:
    """
    stores test 
    """
    def __init__(self, kernel, C, gamma, degree, valid_error, train_time):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma if kernel != "linear" else -1
        self.degree = degree if kernel != "poly" else -1
        self.valid_error = valid_error
        self.train_time = train_time
    
    def get_hypers(self):
        """
        retuns hyper parameters
        """
        match self.kernel:
            case "linear": return self.C
            case "rbf": return self.C, self.gamma
            case "poly": return self.C, self.gamma, self.degree
            case _: raise Exception(f"Kernal value: {self.kernel} does not exist")
    
    def get_param_list(self):
        return [self.kernel, self.C, self.gamma, self.degree, self.valid_error, self.train_time]
    
    def get_param_dict(self):
        return {"kernal": self.kernel, 
                "C": self.C, 
                "gamma": self.gamma, 
                "degree": self.degree, 
                "valid error": self.valid_error, 
                "train time":self.train_time}


# Todo: write functions to find best hyper parameters