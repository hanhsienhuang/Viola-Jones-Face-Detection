import adaboost
import classifiers
import numpy as np

class Cascade:
    def __init__(self):
        self.stages = []

    def __call__(self, x):
        indexes = np.arange(len(x), dtype = int) 
        final_f = np.full(len(x), float("-inf"))
        for c, t in self.stages:
            f = c(x)
            pos_index = f > t
            x = x[pos_index]
            indexes = indexes[pos_index]
            f = f[pos_index]
        final_f[indexes] = f-t
        return final_f

    def predict(self, x):
        return np.sign(self.__call__(x)).astype(int)
    
    def append(self, classifier, threshold):
        self.stages.append((classifier, threshold))

def find_threshold(y, f, tpr):
    index = np.argsort(f)
    sorted_f = f[index]
    sorted_y = y[index]
    tp = 0
    fp = 0
    p = (sorted_y == 1).sum()
    n = len(sorted_y) - p
    for i in range(len(y)-1, -1, -1):
        if sorted_y[i] == 1:
            tp += 1
        else:
            fp += 1

        if i > 0 and sorted_f[i] == sorted_f[i-1]:
            continue

        if tp / p > tpr:
            if i == 0:
                threshold = sorted_f[0] - 1
            elif i == len(y)-1:
                threshold = sorted_f[-1] + 1
            else:
                threshold = (sorted_f[i] + sorted_f[i-1])/2
            return threshold, tp/p, fp/n

def train_single_stage(x, y, d, f):
    weight = np.ones(x.shape[0])
    pos = y==1
    neg = ~pos
    weight[pos] /= 2*pos.sum()
    weight[neg] /= 2*neg.sum()

    classifier = classifiers.Classifier()
    fpr = float("inf")
    
    i = 0
    while fpr >= f:
        i+=1
        print("    step: {}".format(i))
        classifier, weight = adaboost.train_once(x, y, weight, classifier)
        threshold, tpr, fpr = find_threshold(y, classifier(x), d)
    return classifier, threshold, tpr, fpr

def train_cascade(x, y, d, f, Ftarget):
    xp = x[y == 1]
    xn = x[y ==-1]

    F = 1
    cascade = Cascade()
    i = 0
    while F > Ftarget:
        i+=1
        print("stage: {}".format(i))
        yp = np.full(len(xp), 1, dtype=y.dtype)
        yn = np.full(len(xn), -1, dtype=y.dtype)
        x = np.concatenate([xp, xn])
        y = np.concatenate([yp, yn])

        classifier, threshold, tpr, fpr = train_single_stage(x, y, d, f)
        cascade.append(classifier, threshold)
        F *= fpr
        if F > Ftarget:
            false_negative = classifier.predict(xn, threshold) == 1
            xn = xn[false_negative]
    return cascade