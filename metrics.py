import numpy as np

def count_true_false(y, y_pred):
    p = (y==1).sum()
    n = len(y) - p 
    tp = ((y==1) & (y_pred ==1)).sum()
    fn = p - tp
    tn = ((y==-1) & (y_pred ==-1)).sum()
    fp = n - tn
    return tp, tn, fp, fn

def accuracy(y, y_pred):
    tp, tn, fp, fn = count_true_false(y, y_pred)
    return (tp + tn)/ (tp+tn+fp+fn)

def f1_score(y, y_pred):
    tp, tn, fp, fn = count_true_false(y, y_pred)
    precision = tp / (tp + fp)
    tpr = tp / (tp + fn)
    return 2 / (1/tpr + 1/precision)

def tpr_fpr(y, y_pred):
    tp, tn, fp, fn = count_true_false(y, y_pred)
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return tpr, fpr

def roc(y, f):
    indexes = np.argsort(f)
    f = f[indexes]
    y = y[indexes]
    tprs = [0]
    fprs = [0]
    p = (y==1).sum()
    n = len(y) - p
    tp = 0
    fp = 0
    for i in range(len(y)-1, -1, -1):
        yi = y[i]
        fi = f[i]
        if yi == 1:
            tp += 1
        else:
            fp += 1

        if i >0 and fi == f[i-1]:
            continue

        tprs.append(tp/p)
        fprs.append(fp/n)
    return tprs, fprs

def auc(y, f):
    tprs, fprs = roc(y, f)
    area = 0
    for i in range(len(tprs)-1):
        xi, xip1 = fprs[i], fprs[i+1]
        yi, yip1 = tprs[i], tprs[i+1]
        area += (xip1-xi) * (yi+yip1) / 2
    return area