import numpy as np

def rect_sum(x, top, left, bot, right):
    return x[:, bot, right] - x[:, bot, left] - x[:, top, right] + x[:, top, left]

def compute_feature(x, t, top, left, bot, right):
    summ = 0
    if t == 0:
        summ -= rect_sum(x, top, (left+right)//2, bot, right)
    elif t == 1:
        summ -= rect_sum(x, top, left, (top+bot)//2, right)
    elif t == 2:
        summ -= rect_sum(x, top, (left*2+right)//3, bot, (left+right*2)//3)
    elif t == 3:
        summ -= rect_sum(x, top, (left+right)//2, (top+bot)//2, right)
        summ -= rect_sum(x, (top+bot)//2, left, bot, (left+right)//2)
    summ *= 2
    summ += rect_sum(x, top, left, bot, right)
    return summ

features_types = [(1, 2), (2,1), (1,3), (2,2)]
def get_features(x): #data: dim (N, w, h)
    n, h, w = x.shape
    y = np.cumsum(np.cumsum(x, 1), 2)
    x = np.zeros((n, h+1, w+1), dtype= y.dtype)
    x[:, 1:, 1:] = y
    features = []
    index_to_feature = []
    for t, (f_h, f_w) in enumerate(features_types):
        for top in range(h):
            for bot in range(top+f_h, h+1, f_h):
                for left in range(w):
                    for right in range(left+f_w, w+1, f_w):
                        features.append(compute_feature(x, t, top, left, bot, right))
                        index_to_feature.append((t,top,left,bot,right))
    return np.stack(features, -1), index_to_feature
