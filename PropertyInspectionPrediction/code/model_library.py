import pandas as pd
import numpy as np
import cPickle as pickle
import os

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
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

global trials_counter

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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
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
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = ExtraTreesRegressor(n_estimators=model_param['n_estimators'])
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # GBRT regression
    if model_list.count('gbf') > 0:
        model_type = 'gbf'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            model = GradientBoostingRegressor(n_estimators=model_param['n_estimators'])
            if type(x_train) != np.ndarray:
                model.fit( x_train.toarray(), y_train )
                pred_val = model.predict( x_test.toarray() )
            else:
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # xgboost
    if model_list.count('xgboost') > 0:
        model_type = 'xgboost'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = config.best_param[model_type]
            params = model_param
            num_rounds = model_param['num_rounds']
            #create a train and validation dmatrices
            xgtrain = xgb.DMatrix(x_train, label=y_train)
            xgval = xgb.DMatrix(x_test, label=y_test)

            #train using early stopping and predict
            watchlist = [(xgtrain, "train"),(xgval, "val")]
            #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
            pred_val = model.predict( xgval, ntree_limit=model.best_iteration )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    if model_list.count('xgboost-art') > 0:
        model_type = 'xgboost-art'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s trainning..." % model_type
            model_param = config.best_param[model_type]
            params = model_param
            num_rounds = model_param['num_rounds']
            offset = int(model_param['valid_size'] * y_train.shape[0])
            if type(x_train) != np.ndarray:
                x_train = x_train.toarray()
                x_test = x_test.toarray()
            xgtrain = xgb.DMatrix(x_train[offset:, :], label=y_train[offset:])
            xgval = xgb.DMatrix(x_train[:offset, :], label=y_train[:offset])
            watchlist = [(xgtrain, "train"), (xgval, "val")]
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = model_param['early_stopping_rounds'])

            xgtest = xgb.DMatrix(x_test)
            pred_val1 = model.predict(xgtest, ntree_limit=model.best_iteration)

            # reverse train, and log label
            x_train_tmp = x_train[::-1, :]
            y_train_tmp = np.log(y_train[::-1])
            xgtrain = xgb.DMatrix(x_train_tmp[offset:, :], label=y_train_tmp[offset:])
            xgval = xgb.DMatrix(x_train_tmp[:offset, :], label=y_train_tmp[:offset])
            watchlist = [(xgtrain, "train"), (xgval, "val")]
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = model_param['early_stopping_rounds'])

            pred_val2 = model.predict(xgtest, ntree_limit=model.best_iteration)
            pred_val = pred_val1*1.5 + pred_val2*8.5
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
                print "Gen pred for (iter%d, fold%d, %s) cross validation" %(iter, fold, feat)
                with open("%s/iter%d/fold%d/train.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_train, y_train] = pickle.load(f)
                with open("%s/iter%d/fold%d/valid.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_val, y_val] = pickle.load(f)
                path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
                train_model(path, x_train, y_train, x_val, y_val, feat)

    # load feat, train/test
    for feat in feat_names:
        print "Gen pred for (%s) all test data" %(feat)
        with open("%s/all/train.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
            [x_train, y_train] = pickle.load(f)
        with open("%s/all/test.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
            [x_test, y_test] = pickle.load(f)
        path = "%s/all" %(config.data_folder)
        train_model(path, x_train, y_train, x_test, y_test, feat)

def hyperopt_wrapper(param, model_type, feat):
    global trials_counter
    trials_counter += 1
    gini_cv_mean, gini_cv_std = hyperopt_obj(param, model_type, feat, trials_counter)
    return -gini_cv_mean


def hyperopt_obj(model_param, model_type, feat, trials_counter):
    ######
    # approach different model
    # Deep Learning Model
    gini_cv = np.zeros((config.kiter, config.kfold), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            # load data
            path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
            with open("%s/train.%s.feat.pkl" %(path, feat), 'rb') as f:
                [x_train, y_train] = pickle.load(f)
            with open("%s/valid.%s.feat.pkl" %(path, feat), 'rb') as f:
                [x_test, y_test] = pickle.load(f)
            pred_val = hyperopt_library(model_type, model_param, x_train, y_train, x_test)
            # save the pred for cross validation
            pred_file = "%s/%s_%s.pred.%d.pkl" %(path, feat, model_type, trials_counter)
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)

            gini_cv[iter, fold] = Gini(y_test, pred_val)

    gini_cv_mean = np.mean(gini_cv)
    gini_cv_std = np.std(gini_cv)
    print "Mean %f, std %f" % (gini_cv_mean, gini_cv_std)

    # save the pred for train/test
    # load data
    path = "%s/all" %(config.data_folder)
    with open("%s/train.%s.feat.pkl" %(path, feat), 'rb') as f:
        [x_train, y_train] = pickle.load(f)
    with open("%s/test.%s.feat.pkl" %(path, feat), 'rb') as f:
        [x_test, y_test] = pickle.load(f)
    pred_val = hyperopt_library(model_type, model_param, x_train, y_train, x_test)
    # save the pred for cross validation
    pred_file = "%s/%s_%s.pred.%d.pkl" %(path, feat, model_type, trials_counter)
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_val, f, -1)

    return gini_cv_mean, gini_cv_std
    ######

def hyperopt_library(model_type, model_param, x_train, y_train, x_test):
    # training
    if model_type.count('dnn') > 0:
        print "%s training..." % model_type
        model = deep_model()
        model.fit(x_train, y_train, nb_epoch=2, batch_size=16)
        pred_val = model.predict( x_test, batch_size=16 )
        pred_val = pred_val.reshape( pred_val.shape[0] )

    # Nearest Neighbors
    if model_type.count('knn') > 0:
        print "%s training..." % model_type
        n_neighbors = model_param['n_neighbors']
        weights = model_param['weights']
        model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # linear regression
    if model_type.count('linear') > 0:
        print "%s training..." % model_type
        model = LinearRegression()
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # logistic regression
    if model_type.count('logistic') > 0:
        print "%s training..." % model_type
        model = LogisticRegression()
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # SVM regression
    if model_type.count('svr') > 0:
        print "%s training..." % model_type
        model = SVR(C=model_param['C'], epsilon=model_param['epsilon'])
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # rank SVM
    if model_type.count('ranksvm') > 0:
        print "%s training..." % model_type
        model = RankSVM().fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # random forest regression
    if model_type.count('rf') > 0:
        print "%s training..." % model_type
        model = RandomForestRegressor(n_estimators=model_param['n_estimators'])
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # extra tree regression
    if model_type.count('extratree') > 0:
        print "%s training..." % model_type
        model = ExtraTreesRegressor(n_estimators=model_param['n_estimators'])
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # GBRT regression
    if model_type.count('gbf') > 0:
        print "%s training..." % model_type
        model = GradientBoostingRegressor(n_estimators=model_param['n_estimators'])
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )

    # xgboost
    if model_type.count('xgboost') > 0:
        print "%s training..." % model_type
        params = model_param
        num_rounds = model_param['num_rounds']
        #create a train and validation dmatrices
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        xgval = xgb.DMatrix(x_test)

        #train using early stopping and predict
        watchlist = [(xgtrain, "train")]
        #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
        model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
        pred_val = model.predict( xgval, ntree_limit=model.best_iteration )

    if model_type.count('xgboost-art') > 0:
        print "%s trainning..." % model_type
        params = model_param
        num_rounds = model_param['num_rounds']
        offset = int(model_param['valid_size'] * y_train.shape[0])
        xgtrain = xgb.DMatrix(x_train[offset:, :], label=y_train[offset:])
        xgval = xgb.DMatrix(x_train[:offset, :], label=y_train[:offset])
        watchlist = [(xgtrain, "train"), (xgval, "val")]
        model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = 100)

        xgtest = xgb.DMatrix(x_test)
        pred_val1 = model.predict(xgtest, ntree_limit=model.best_iteration)

        # reverse train, and log label
        x_train_tmp = x_train[::-1, :]
        y_train_tmp = np.log(y_train[::-1])
        xgtrain = xgb.DMatrix(x_train_tmp[offset:, :], label=y_train_tmp[offset:])
        xgval = xgb.DMatrix(x_train_tmp[:offset, :], label=y_train_tmp[:offset])
        watchlist = [(xgtrain, "train"), (xgval, "val")]
        model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = 100)

        pred_val2 = model.predict(xgtest, ntree_limit=model.best_iteration)
        pred_val = pred_val1*1.5 + pred_val2*8.5

    return pred_val

def hyperopt_main():
    feat_names = config.feat_names
    model_list = config.model_list
    for feat in feat_names:
        for model in model_list:
            model_param = config.param_spaces[model]
            global trials_counter
            trials_counter = 0
            trials = Trials()
            obj = lambda p: hyperopt_wrapper(p, model, feat)
            best_params = fmin(obj, model_param, algo=tpe.suggest, trials=trials, max_evals=config.hyper_max_evals)
            print best_params

if __name__ == '__main__':
    start_time = time.time()

    # write your code here
    # apply different model on different feature, generate model library

    flag = sys.argv[1]
    if flag == "train":
        ## generate pred by best params
        one_model()

    if flag == "hyperopt":
        ## hyper parameter search
        hyperopt_main()



    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
