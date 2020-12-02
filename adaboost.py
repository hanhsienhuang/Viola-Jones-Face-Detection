import decision_stump
import numpy as np
import classifiers

def train_once(x, y, weight, classifier):
    base_hypothesis, epsilon = decision_stump.train(x, y, weight)
    prediction = base_hypothesis(x)

    multiplier = epsilon/(1 - epsilon)
    weight = weight * multiplier ** (y == prediction).astype(int)
    weight = weight / np.sum(weight)

    alpha = 0.5 * np.log(1/epsilon -1)
    classifier.append(base_hypothesis, alpha)

    return classifier, weight

def train(x, y, round):
    classifier = classifiers.Classifier()
    weight = np.ones(x.shape[0])
    pos = y == 1
    weight[pos] /= 2*pos.sum()
    neg = ~pos
    weight[neg] /= 2*neg.sum()
    for _ in range(round):
        classifier, weight = train_once(x, y, weight, classifier)
        base = classifier[-1][0]
        print(base.get_index(), base.get_theta(), base.get_parity())
    return classifier

