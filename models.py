from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np

def make_feat_matrix(full_dict):
    outerList = []
    for year in full_dict:
        year_dict = full_dict[year]
        for team in year_dict:
            innerList = []
            innerList.append(abs (year_dict[team]['pct'] - .5))
            innerList.append(year_dict[team]['opp_ypa'])
            innerList.append(year_dict[team]['opp_ypc'])

            outerList.append(innerList)

    return np.array(outerList)

def make_label_arr(full_dict):
    outerList = []
    dict = {.75 : 1, .875 : 1, 1 : 1, 2 : 2, 2.286 : 2, 2.667 : 2, 4 : 3, 8 : 4, 16 : 5, 32 : 6}
    for year in full_dict:
        year_dict = full_dict[year]
        for team in year_dict:
            outerList.append(dict[year_dict[team]['outcome']])
            
    return np.array(outerList)

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