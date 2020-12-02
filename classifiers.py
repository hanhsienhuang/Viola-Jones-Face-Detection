import numpy as np

class Base:
    def __init__(self, index, theta, parity):
        self.index = index
        self.theta = theta
        self.parity = parity
    
    def __call__(self, x):
        return np.sign(self.parity * (self.theta - x[:, self.index]))

    def get_index(self):
        return self.index
    
    def get_parity(self):
        return self.parity
    
    def get_theta(self):
        return self.theta

class Classifier:
    def __init__(self, params = None):
        self.parameters = []
        if params is not None:
            self.parameters += params

    def __call__(self, x):
        return sum([alpha * base(x) for base, alpha in self.parameters])

    def predict(self, x, threshold = 0):
        return np.sign(self.__call__(x) - threshold).astype(int)

    def append(self, base, alpha):
        self.parameters.append((base, alpha))
    
    def __getitem__(self, key):
        return self.parameters[key]
    
    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)
