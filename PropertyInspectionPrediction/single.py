import pandas as pd
import numpy as np

from sklearn import preprocessing

import xgboost as xgb

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from utils import *

def feature_engineer():
    train = pd.read_csv('data/train.csv', header=0)
    test = pd.read_csv('data/test.csv', header=0)

    labels = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)

    idx = test.Id

    train = np.array(train)
    test = np.array(test)

    # label encode the categorical variables
    for i in range(train.shape[1]):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

    return train, labels, test, idx


def train_model(train, labels, test, idx):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 5
    params["subsample"] = 0.75
    params["colsample_bytree"] = 0.85
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 8


    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    kfold = 10
    kiter = 2
    train_subsets, labels_subsets = cv_split(train, labels, kfold, kiter)

    best_score = 0
    best_model = None
    for i in range(kiter):
        gini_sum = 0
        for f in range(kfold):
            #create a train and validation dmatrices
            xgtrain = xgb.DMatrix(train_subsets[i][f][0], label=labels_subsets[i][f][0])
            xgval = xgb.DMatrix(train_subsets[i][f][1], label=labels_subsets[i][f][1])

            #train using early stopping and predict
            watchlist = [(xgtrain, "train"),(xgval, "val")]
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)

            pred_val = model.predict(xgval)
            gini_f = Gini(labels_subsets[i][f][1], pred_val)
            gini_sum += gini_f
        print "Iter %d, Gini Mean is %f" %(i, gini_sum/kfold)

    write_submission(idx, pred, 'single_best.csv')
    return pred




def hyperopt_obj(param, train, label_t, val, label_v):
    gini_score = train_one_model(param, train, label_t, val, label_v)
    return {
        'loss': gini_score,
        'status': STATUS_OK,
    }

def hyperopt_space():
    space = hp.choice('model_type', [
        {
            'type': 'xgboost',
            'booster': hp.choice('booster', ['reg:gbtree', 'reg:gblinear']),
        },
    ])
    return space

def param_opt():
    train, labels, test, idx = feature_engineer()

    kfold = 2
    kiter = 1

    train_subsets_k, label_subsets_k = cv_split(train, labels, kfold, kiter)

    train_sub = train_subsets_k[0][0][0]
    label_t = label_subsets_k[0][0][0]
    val_sub = train_subsets_k[0][0][1]
    label_v = label_subsets_k[0][0][1]
    param_space = {
        'model_type': 'xgboost',
        'booster': hp.choice('booster', ['gbtree', 'gblinear']),
        'objective': 'reg:linear',
        'eta': 0.005,
        'min_child_weight': 5,
        'subsample': hp.uniform('subsample', 0.7, 0.9),
        'silent': 0,
        'max_depth': hp.quniform('max_depth', 6, 8, 1)
    }

    obj = lambda param: hyperopt_obj(param, train_sub, label_t, val_sub, label_v)
    trials = Trials()
    best_model = fmin(obj,
        space = param_space,
        algo = tpe.suggest,
        max_evals = 100,
        trials = trials)
    return best_model

if __name__ == '__main__':
    #print 'single model, hyperopt parameter'
    #best_param = param_opt()

    print 'single model'
    train, labels, test, idx = feature_engineer()
    train_model(train, labels, test, idx)
