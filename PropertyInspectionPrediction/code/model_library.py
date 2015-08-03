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


def train_model(path, x_train, y_train, x_test, y_test, model_type, model_param, feat):
    ######
    # approach different model
    # Deep Learning Model
    if model_type.count('dnn') > 0:
        model = deep_model()
        model.fit(x_train, y_train, nb_epoch=2, batch_size=16)
        pred_val = model.predict( x_test, batch_size=16 )
        pred_val = pred_val.reshape( pred_val.shape[0] )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # Nearest Neighbors
    if model_type.count('knn') > 0:
        n_neighbors = model_param['n_neighbors']
        weights = model_param['weights']
        model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # linear regression
    if model_type.count('linear') > 0:
        model = LinearRegression()
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # logistic regression
    if model_type.count('logistic') > 0:
        model = LogisticRegression()
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # SVM regression
    if model_type.count('svr') > 0:
        model = SVR(C=model_param['C'], epsilon=model_param['epsilon'])
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # rank SVM
    if model_type.count('ranksvm') > 0:
        model = RankSVM().fit( x_train, y_train )
        pred_val = model.predict( x_test )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # random forest regression
    if model_type.count('rf') > 0:
        model = RandomForestRegressor(n_estimators=model_param['n_estimators'])
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # extra tree regression
    if model_type.count('extratree') > 0:
        model = ExtraTreesRegressor(n_estimators=model_param['n_estimators'])
        model.fit( x_train, y_train )
        pred_val = model.predict( x_test )
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)

    # xgboost
    if model_type.count('xgboost') > 0:
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
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)
    ######
    return best_model, model_type

def predict(best_model, model_type, test, idx):
    if model_type.count('xgboost') > 0:
        best_model.save_model('model/single_'+model_type+'.mod')
        xgtest = xgb.DMatrix(test)
        pred = best_model.predict(xgtest)
    elif model_type.count('dnn') > 0:
        pred = best_model.predict(test, batch_size=16)
        pred = pred.reshape( pred.shape[0] )
    else:
        pred = best_model.predict(test)
    write_submission(idx, pred, 'single_' + model_type + '.csv')

def one_model(param, model_type):
    model_param = param.param_space[model_type]
    # load feat, cross validation
    cv_score = []
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            for feat in feat_names:
                with open("%s/iter%d/fold%d/train.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_train, y_train] = pickle.load(f)
                with open("%s/iter%d/fold%d/valid.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_val, y_val] = pickle.load(f)
                path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
                score = train_model(path, x_train, y_train, x_val, y_val, model_type, model_param, feat)
                cv_score.append(score)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python single.py [rf, extratree, svr, ranksvm, xgboost, linear, logistic, knn, dnn]'
        exit(1)
    start_time = time.time()

    # write your code here
    # apply different model on different feature, generate model library

    # load data
    train = pd.read_csv(config.origin_train_path, header=0)
    test = pd.read_csv(config.origin_test_path, header=0)

    with open(config.feat_names_file, 'rb') as f:
        feat_names = pickle.load(f)



    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
