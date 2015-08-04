import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import csv
import time

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, pyll

from utils import *


def xgboost_pred(param, train,labels,test, weight):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 9

    plst = list(params.items())

    #Using 5000 rows for early stopping.
    offset = 4000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
    #xgtrain = xgb.DMatrix(train, label=labels)
    #xgval = xgb.DMatrix(train, label=labels)

    #train using early stopping and predict
    watchlist = [(xgtrain, "train"),(xgval, "val")]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    #model = xgb.train(plst, xgtrain, 800)
    pred1_val = model.predict(xgval,  ntree_limit=model.best_iteration)
    print Gini(labels[:offset], pred1_val)
    #print Gini(labels, pred1_val)
    preds1 = model.predict(xgtest)


    #reverse train and labels and use different 5k for early stopping.
    # this adds very little to the score but it is an option if you are concerned about using all the data.
    train = train[::-1,:]
    labels = np.log(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
    #xgtrain = xgb.DMatrix(train, label=labels)
    #xgval = xgb.DMatrix(train, label=labels)


    watchlist = [(xgtrain, "train"),(xgval, "val")]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
#    model = xgb.train(plst, xgtrain, 800)
    pred2_val = model.predict(xgval, ntree_limit=model.best_iteration)
    print Gini(labels[:offset], pred2_val)
    #print Gini(labels, pred2_val)
    preds2 = model.predict(xgtest)


    #combine predictions
    #since the metric only cares about relative rank we don"t need to average
    preds = preds1*weight + preds2*(1-weight)
    return preds

# test has labels value
def ensemble_obj(param, train_tmp, test_tmp):
    print 'xgboost, two time ensemble'

    train = train_tmp.copy()
    test = test_tmp.copy()

    labels = train.Hazard
    train.drop("Hazard", axis=1, inplace=True)

    y_true = test.Hazard
    test.drop("Hazard", axis=1, inplace=True)

    train_s = train
    test_s = test


    train_s.drop("T2_V10", axis=1, inplace=True)
    train_s.drop("T2_V7", axis=1, inplace=True)
    train_s.drop("T1_V13", axis=1, inplace=True)
    train_s.drop("T1_V10", axis=1, inplace=True)

    test_s.drop("T2_V10", axis=1, inplace=True)
    test_s.drop("T2_V7", axis=1, inplace=True)
    test_s.drop("T1_V13", axis=1, inplace=True)
    test_s.drop("T1_V10", axis=1, inplace=True)

    test_ind = test.index

    train_s = np.array(train_s)
    test_s = np.array(test_s)

    # label encode the categorical variables
    for i in range(train_s.shape[1]):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
        train_s[:,i] = lbl.transform(train_s[:,i])
        test_s[:,i] = lbl.transform(test_s[:,i])

    train_s = train_s.astype(float)
    test_s = test_s.astype(float)


    preds1 = xgboost_pred(param, train_s,labels,test_s, param['weight_inter'])

    #model_2 building

    train = train.T.to_dict().values()
    test = test.T.to_dict().values()

    vec = DictVectorizer()
    train = vec.fit_transform(train)
    test = vec.transform(test)

    preds2 = xgboost_pred(param, train,labels,test, param['weight_inter'])


    weight = param['weight']
    preds = weight * (preds1**param['pow_weight']) + (1-weight) * (preds2**param['pow_weight'])

    score = Gini(y_true, preds)
    writer.writerow( [score] + list(param.values()) )
    o_f.flush()
    return 1.0/score


def param_opt(param, train, test):
    writer.writerow( ['gini'] + list(param.keys()) )
    o_f.flush()

    obj = lambda param: ensemble_obj(param, train, test)
    trials = Trials()
    best_model = fmin(obj,
        space = param,
        algo = tpe.suggest,
        max_evals = 100,
        trials = trials)
    return best_model

def xgb_split(kfold):
    #load train and test
    train  = pd.read_csv("data/train.csv", index_col=0)
    test  = pd.read_csv("data/test.csv", index_col=0)

    n_sample = len(train.index) / kfold
    data_split_sets = []
    #for k in range(kfold):
    train_s = train[:n_sample].copy()
    test_s = train[n_sample:].copy()
    data_split_sets.append( [train_s, test_s] )

    train_t = train[n_sample:].copy()
    test_t = train[:n_sample].copy()
    data_split_sets.append( [train_t, test_t] )
    return train, test, data_split_sets

if __name__ == '__main__':
    start_time = time.time()
    kfold = 2
    train, test, data_split_sets = xgb_split(kfold)

    param = {
        #'offset': pyll.scope.int(hp.quniform('offset', 4000, 10000, 1000)),
        'weight': 0.463, #hp.quniform('weight', 0.45, 0.5, 0.001),
        'weight_inter': 1.5, #hp.quniform('weight_inter', 1.4, 1.7, 0.01),
        'pow_weight': hp.quniform('pow_weight', 0, 1, 0.01),
        #'early_stopping_rounds': pyll.scope.int(hp.quniform('early_stopping_rounds', 50, 150, 10)),
        #'objective': 'reg:linear',
        #'eta': hp.quniform('eta', 0.001, 0.01, 0.001),
        #'gamma': hp.quniform('gamma', 0, 10, 0.1),
        #'max_depth': pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
        #'min_child_weight': hp.quniform('min_child_weight', 0, 10, 0.1),
        #'max_delta_step': pyll.scope.int(hp.quniform('max_delta_step', 1, 10, 1)),
        #'subsample': hp.quniform('subsample', 0.7, 1, 0.01),
        #'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.01),
        #'lambda': hp.quniform('lambda', 0, 1, 0.1),
        #'alpha': hp.quniform('alpha', 0, 1, 0.1),
        #'lambda_bias': hp.quniform('lambda_bias', 0, 1, 0.1),
    }

    for k in range(kfold):
        o_f = open('hyperlogs/xgboost_pow_' + str(k) + '.log', 'wb')
        writer = csv.writer(o_f)
        train_s = data_split_sets[k][0]
        test_s = data_split_sets[k][1]
        best = param_opt(param, train_s, test_s)
        print best
        o_f.close()

    ##generate solution
    #preds = pd.DataFrame({"Id": idx, "Hazard": preds})
    #preds = preds.set_index("Id")
    #preds.to_csv("xgboost_benchmark_kk.csv")

    print 'Second elapsed ', (time.time() - start_time)/1000
