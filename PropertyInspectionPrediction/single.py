import pandas as pd
import numpy as np

from sklearn import preprocessing

import xgboost as xgb

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from utils import *


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
