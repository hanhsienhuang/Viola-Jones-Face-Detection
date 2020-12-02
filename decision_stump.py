import numpy as np
import classifiers

def train(x, y, weight):
    m, d = x.shape
    loss = float("inf")
    opt = None

    indexes = np.argsort(x, axis=0)
    for i in range(d):
        index = indexes[:,i]
        x_tmp = x[index, i]
        y_tmp = y[index]
        w_tmp = weight[index]

        loss_tmp = np.sum(w_tmp[y_tmp==1])
        if loss_tmp < loss:
            loss = loss_tmp
            theta = x_tmp[0] - 1 
            opt = (i, theta, 1)
        if (1-loss_tmp) < loss:
            loss = 1-loss_tmp
            theta = x_tmp[0] - 1 
            opt = (i, theta, -1)

        for k in range(m):
            loss_tmp -= w_tmp[k] * y_tmp[k]
            if(k < m-1 and x_tmp[k] == x_tmp[k+1]):
                continue

            if loss_tmp < loss:
                loss = loss_tmp
                if k == m-1:
                    theta = x_tmp[k] + 1
                else:
                    theta = (x_tmp[k] + x_tmp[k+1])/2
                opt = (i, theta, 1)

            if (1-loss_tmp) < loss:
                loss = 1-loss_tmp
                if k == m-1:
                    theta = x_tmp[k] + 1
                else:
                    theta = (x_tmp[k] + x_tmp[k+1])/2
                opt = (i, theta, -1)
    return classifiers.Base(*opt), loss

