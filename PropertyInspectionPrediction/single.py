import pandas as pd
import numpy as np

from sklearn import preprocessing

import xgboost as xgb
from rankSVM import RankSVM

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from utils import *
from ml_metrics import *

from param import ParamConfig

import time

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


def train_model(param, train, labels, test, idx):
    xgtest = xgb.DMatrix(test)

    kfold = param.kfold
    kiter = param.kiter
    train_subsets, labels_subsets = cv_split(train, labels, kfold, kiter)

    if param.model_type == 'RankSVM':
        print 'RankSVM'
        best_score = 0
        best_model = None
        for i in range(kiter):
            gini_sum = 0
            for f in range(kfold):
                model = RankSVM().fit( train_subsets[i][f][0], labels_subsets[i][f][0] )
                pred_val = model.predict( train_subsets[i][f][1] )
                gini_f = Gini( labels_subsets[i][f][1], pred_val )
                gini_sum += gini_f
                if gini_f > best_score:
                    best_score = gini_f
                    best_model = model
            print "Iter %d, Gini Mean is %f" %(i, float(gini_sum)/kfold)

        best_model.save_model('model/single_ranksvm.mod')
        pred = best_model.predict(xgtest)
        write_submission(idx, pred, 'single_ranksvm.csv')


    if param.model_type == 'xgboost':
        params = {}
        params["objective"] = "rank:pairwise"
        params["eta"] = 0.005
        params["min_child_weight"] = 5
        params["subsample"] = 0.75
        params["colsample_bytree"] = 0.85
        params["scale_pos_weight"] = 1.0
        params["silent"] = 1
        params["max_depth"] = 8


        num_rounds = 10000

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
                #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)

                pred_val = model.predict(xgval)
                gini_f = Gini(labels_subsets[i][f][1], pred_val)
                #print "Iter %d, fold %d, Gini Mean is %f" %(i, f, gini_f)
                gini_sum += gini_f

                if gini_f > best_score:
                    best_score = gini_f
                    best_model = model
            print "Iter %d, Gini Mean is %f" %(i, float(gini_sum)/kfold)

        best_model.save_model('model/single_best.mod')
        pred = best_model.predict(xgtest)
        write_submission(idx, pred, 'single_best.csv')


if __name__ == '__main__':
    start_time = time.time()

    param = ParamConfig()
    print 'single model'
    train, labels, test, idx = feature_engineer()
    train_model(param, train, labels, test, idx)

    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
