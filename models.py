from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np

def train_test_eval_linear(xs, ys):
    yearAccs = []
    yearPlayoffCorrect = []
    for i in range (21):
        model = LinearRegression()
        rangelow = i * 32
        rangehi = (i * 32) + 32
        xtrain = np.concatenate([xs[:rangelow], xs[rangehi:]], axis=0)
        ytrain = np.concatenate([ys[:rangelow], ys[rangehi:]], axis=0)
        x_test = xs[rangelow:rangehi]
        y_test = ys[rangelow:rangehi]
        model.fit(np.array(xtrain), np.array(ytrain))
        y_pred = model.predict(x_test)
        # print(y_pred)
        avg = 0
        num_playoff_correct = 0
        num_false_negs = 0
        for i in range (0, 32):
            if y_test[i] > 1:
                if y_pred[i] > 1.5:
                    num_playoff_correct += 1
            if y_pred[i] > 1.5:
                if y_test[i] == 1:
                    num_false_negs += 1
            diff = abs(y_pred[i] - y_test[i])
            avg += diff

        yearPlayoffCorrect.append(num_playoff_correct)

        yearAccs.append(avg / 32)




    return yearAccs

def train_test_eval_trees(xs, ys):
    yearAccs = []
    yearPlayoffCorrect = []
    for i in range (21):
        model = RandomForestClassifier(class_weight='balanced')
        rangelow = i * 32
        rangehi = (i * 32) + 32
        xtrain = np.concatenate([xs[:rangelow], xs[rangehi:]], axis=0)
        ytrain = np.concatenate([ys[:rangelow], ys[rangehi:]], axis=0)
        x_test = xs[rangelow:rangehi]
        y_test = ys[rangelow:rangehi]
        model.fit(np.array(xtrain), np.array(ytrain))
        y_pred = model.predict(x_test)
        avg = 0
        num_playoff_correct = 0
        num_false_negs = 0
        for i in range (0, 32):
            if y_test[i] > 1:
                if y_pred[i] > 1.5:
                    num_playoff_correct += 1
            if y_pred[i] > 1.5:
                if y_test[i] == 1:
                    num_false_negs += 1
            diff = abs(y_pred[i] - y_test[i])
            avg += diff

        yearPlayoffCorrect.append(num_playoff_correct)

        yearAccs.append(avg / 32)

    return yearPlayoffCorrect