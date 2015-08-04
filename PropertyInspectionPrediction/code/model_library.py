import pandas as pd
import numpy as np
import cPickle as pickle
import os

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

from param import config

import time
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adagrad
def deep_model():
    model = Sequential()
    model.add(Dense(33, 20, init='uniform', activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(20, 10, init='uniform', activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(10, 1, init='uniform', activation='linear'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    #model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
    #score = model.evaluate(X_test, y_test, batch_size=16)
    return model


def train_model(path, x_train, y_train, x_test, y_test, feat):

    model_list = config.model_list
    ######
    # approach different model
    # Deep Learning Model
    if model_list.count('dnn') > 0:
        model_type = 'dnn'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = deep_model()
            model.fit(x_train, y_train, nb_epoch=2, batch_size=16)
            pred_val = model.predict( x_test, batch_size=16 )
            pred_val = pred_val.reshape( pred_val.shape[0] )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # Nearest Neighbors
    if model_list.count('knn') > 0:
        model_type = 'knn'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            n_neighbors = model_param['n_neighbors']
            weights = model_param['weights']
            model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # linear regression
    if model_list.count('linear') > 0:
        model_type = 'linear'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = LinearRegression()
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # logistic regression
    if model_list.count('logistic') > 0:
        model_type = 'logistic'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = LogisticRegression()
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # SVM regression
    if model_list.count('svr') > 0:
        model_type = 'svr'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = SVR(C=model_param['C'], epsilon=model_param['epsilon'])
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # rank SVM
    if model_list.count('ranksvm') > 0:
        model_type = 'ranksvm'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = RankSVM().fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # random forest regression
    if model_list.count('rf') > 0:
        model_type = 'rf'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = RandomForestRegressor(n_estimators=model_param['n_estimators'])
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print 'Done!'

    # extra tree regression
    if model_list.count('extratree') > 0:
        model_type = 'extratree'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = ExtraTreesRegressor(n_estimators=model_param['n_estimators'])
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # xgboost
    if model_list.count('xgboost') > 0:
        model_type = 'xgboost'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            params = model_param
            num_rounds = model_param['num_rounds']
            #create a train and validation dmatrices
            xgtrain = xgb.DMatrix(x_train, label=y_train)
            xgval = xgb.DMatrix(x_test, label=y_test)
            print x_test.shape[0]
            print len(y_test)

            #train using early stopping and predict
            watchlist = [(xgtrain, "train"),(xgval, "val")]
            #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
            pred_val = model.predict( xgval, ntree_limit=model.best_iteration )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"
    ######

def one_model():
    # load feat names
    feat_names = config.feat_names

    # load feat, cross validation
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            for feat in feat_names:
                with open("%s/iter%d/fold%d/train.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_train, y_train] = pickle.load(f)
                with open("%s/iter%d/fold%d/valid.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_val, y_val] = pickle.load(f)
                path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
                train_model(path, x_train, y_train, x_val, y_val, feat)

    # load feat, train/test
    for feat in feat_names:
        with open("%s/all/train.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
            [x_train, y_train] = pickle.load(f)
        with open("%s/all/test.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
            [x_test, y_test] = pickle.load(f)
        path = "%s/all" %(config.data_folder)
        train_model(path, x_train, y_train, x_test, y_test, feat)


if __name__ == '__main__':
    start_time = time.time()

    # write your code here
    # apply different model on different feature, generate model library

    one_model()



    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
