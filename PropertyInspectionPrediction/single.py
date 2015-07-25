import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import neighbors

import xgboost as xgb
from rankSVM import RankSVM

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from utils import *
from ml_metrics import *

from param import ParamConfig

import time
import sys

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


def train_model(model_param, param, train, labels, flag='train'):
    kfold = param.kfold
    kiter = param.kiter
    train_subsets, labels_subsets = cv_split(train, labels, kfold, kiter)

    best_score = 0
    best_model = None
    model_type = param.model_type
    #print model_type
    gini_sum = 0
    for i in range(kiter):
        for f in range(kfold):
            #print "Iter %d, fold %d, start train" %(i, f)

            ######
            # approach different model
            # Nearest Neighbors
            if model_type.count('knn') > 0:
                n_neighbors = model_param['n_neighbors']
                weights = model_param['weights']
                model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                model.fit( train_subsets[i][f][0], labels_subsets[i][f][0] )

            # linear regression
            if model_type.count('linear') > 0:
                model = LinearRegression()
                model.fit( train_subsets[i][f][0], labels_subsets[i][f][0] )

            # logistic regression
            if model_type.count('logistic') > 0:
                model = LogisticRegression()
                model.fit( train_subsets[i][f][0], labels_subsets[i][f][0] )

            # SVM regression
            if model_type.count('svr') > 0:
                model = SVR(C=model_param['C'], epsilon=model_param['epsilon'])
                model.fit( train_subsets[i][f][0], labels_subsets[i][f][0] )

            # rank SVM
            if model_type.count('ranksvm') > 0:
                model = RankSVM().fit( train_subsets[i][f][0], labels_subsets[i][f][0] )

            # random forest regression
            if model_type.count('rf') > 0:
                model = RandomForestRegressor(n_estimators=model_param['n_estimators'])
                model.fit( train_subsets[i][f][0], labels_subsets[i][f][0] )

            # extra tree regression
            if model_type.count('extratree') > 0:
                model = ExtraTreesRegressor(n_estimators=model_param['n_estimators'])
                model.fit( train_subsets[i][f][0], labels_subsets[i][f][0] )

            # xgboost
            if model_type.count('xgboost') > 0:
                params = model_param
                num_rounds = model_param['num_rounds']
                #create a train and validation dmatrices
                xgtrain = xgb.DMatrix(train_subsets[i][f][0], label=labels_subsets[i][f][0])
                xgval = xgb.DMatrix(train_subsets[i][f][1], label=labels_subsets[i][f][1])

                #train using early stopping and predict
                watchlist = [(xgtrain, "train"),(xgval, "val")]
                #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=RMSE)
                pred_val = model.predict( xgval, ntree_limit=model.best_iteration )
            ######

            if model_type.count('xgboost') == 0:
                pred_val = model.predict( train_subsets[i][f][1] )
            gini_f = Gini( labels_subsets[i][f][1], pred_val )
            #print "Iter %d, fold %d,  Gini Mean is %f" %(i, f, gini_f)
            gini_sum += gini_f
            if gini_f > best_score:
                best_score = gini_f
                best_model = model
        #print "Iter %d, Gini Mean is %f" %(i, float(gini_sum)/kfold)
    print "All Gini Mean is %f" %(float(gini_sum)/(kfold*kiter))
    if flag == 'param':
        return (kfold*kiter)/float(gini_sum)
    return best_model, model_type

def predict(best_model, model_type, test, idx):
    if model_type.count('xgboost') > 0:
        best_model.save_model('model/single_'+model_type+'.mod')
        xgtest = xgb.DMatrix(test)
        pred = best_model.predict(xgtest)
    else:
        pred = best_model.predict(test)
    write_submission(idx, pred, 'single_' + model_type + '.csv')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python single.py [rf, extratree, svr, ranksvm, xgboost, linear, logistic, knn]'
        exit(1)
    start_time = time.time()

    param = ParamConfig()
    param.model_type = sys.argv[1]
    model_param = param.best_param[param.model_type]
    print 'single model'
    train, labels, test, idx = feature_engineer()
    best_model, model_type = train_model(model_param, param, train, labels)
    predict(best_model, model_type, test, idx)

    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
