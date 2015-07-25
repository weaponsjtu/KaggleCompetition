import pandas as pd
import numpy as np

from sklearn import preprocessing

import xgboost as xgb

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from utils import *
from ml_metrics import *
from single import feature_engineer, train_model
from param import ParamConfig

import time
import sys
import csv

def hyperopt_obj(model_param, param, train, labels):
    gini_score = train_model(model_param, param, train, labels, flag='param')

    # record the parameters
    writer.writerow( [gini_score] + list(model_param.values()) )

    return gini_score

def param_opt(param):
    train, labels, test, idx = feature_engineer()

    model_param = param.param_spaces[param.model_type]

    writer.writerow( ['gini'] + list(model_param.keys()) )

    obj = lambda model_param: hyperopt_obj(model_param, param, train, labels)
    trials = Trials()
    best_model = fmin(obj,
        space = model_param,
        algo = tpe.suggest,
        max_evals = 100,
        trials = trials)
    return best_model

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python param_search.py [rf, extratree, svr, ranksvm, xgboost, linear, logistic, knn]'
        exit(1)

    start_time = time.time()
    print 'single model, hyperopt parameter'

    param = ParamConfig()
    param.model_type = sys.argv[1]

    o_f = open( 'hyperlogs/single_' + param.model_type + '.log', 'wb' )
    writer = csv.writer( o_f )

    best_param = param_opt(param)

    print best_param

    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
