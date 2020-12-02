import data_loader
import feature
import adaboost
import numpy as np
import pickle
import matplotlib.pyplot as plt
import visulization
import metrics
import classifiers
import cascade

train_x, train_y, test_x, test_y = data_loader.load()

train_f, i_f = feature.get_features(train_x)
test_selection_index = np.concatenate([range(472), np.random.choice(19572, 2000, replace=False) + 472])
test_x = test_x[test_selection_index]
test_y = test_y[test_selection_index]


try:
    with open("classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
except:
    classifier = adaboost.train(train_f, train_y, 10)
    with open("classifier.pkl", "wb") as f:
        pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

# 0.99, 0.4, 0.01
try:
    with open("cascade.pkl", "rb") as f:
        cascade_classifier = pickle.load(f)
except:
    cascade_classifier = cascade.train_cascade(train_f, train_y, 0.99, 0.4, 0.01)
    with open("cascade.pkl", "wb") as f:
        pickle.dump(cascade_classifier, f, protocol=pickle.HIGHEST_PROTOCOL)


test_f, i_f = feature.get_features(test_x)

f_pred = classifier(test_f)
y_pred = classifier.predict(test_f)
print(metrics.tpr_fpr(test_y, y_pred))
print(metrics.auc(test_y, f_pred))


# Top 10 features
shape = (19,19)
for i, (base, alpha) in enumerate(classifier):
    print(base.index)
    print("Feature {}: theta {:.2f}, alpha {:.2f}".format(i, base.theta, alpha))
    visulization.visualize_feature(shape, i_f[base.index], base.parity, save="feature_{}.png".format(i), show=False)

plt.figure()
for i in [1, 3, 5, 10]:
    c = classifiers.Classifier(classifier[:i])
    f_pred = c(test_f)
    y_pred = c.predict(test_f)
    print("AUC with {} rounds: {:.3f}".format(i, metrics.auc(test_y, f_pred)))
    if i == 10:
        visulization.plot_roc(test_y, f_pred, label = str(i), show=False, save = "ROC.png")
    else:
        visulization.plot_roc(test_y, f_pred, label = str(i), show=False)



f_pred = cascade_classifier(test_f)
y_pred = cascade_classifier.predict(test_f)
print(metrics.tpr_fpr(test_y, y_pred))
print(metrics.auc(test_y, f_pred))

for s in cascade_classifier.stages:
    print(len(s[0]))
#visulization.plot_roc(test_y, f_pred)
