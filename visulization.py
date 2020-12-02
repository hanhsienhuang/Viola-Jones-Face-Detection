import matplotlib.pyplot as plt
import numpy as np
import metrics

def visualize_feature(shape, feature_type, parity, show = True, save = None):
    t, top, left, bot, right = feature_type
    grids = np.full(shape, 0.5)
    grids[top:bot, left:right] = 1
    if t == 0:
        grids[top:bot, (left+right)//2:right] = 0
    elif t == 1:
        grids[(top+bot)//2:bot, left:right] = 0
    elif t == 2:
        grids[top:bot, (left*2+right)//3:(left+right*2)//3] = 0
    elif t == 3:
        grids[top:(top+bot)//2, left:(left+right)//2] = 0
        grids[(top+bot)//2:bot, (left+right)//2:right] = 0
    if parity == -1:
        grids = 1- grids
    plt.figure(figsize=(5,5))
    plt.imshow(grids, cmap = "gray")
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()

def plot_roc(y, f, label = "", show=True, save = None):
    tprs, fprs = metrics.roc(y, f)
    plt.plot(fprs, tprs, label = label)
    plt.gca().set_aspect(1)
    plt.title("ROC")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()