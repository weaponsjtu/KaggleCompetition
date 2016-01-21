import pandas as pd
import numpy as np
from sklearn import cross_validation, preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import cPickle as pickle

import time, math


### predict whether this user will view vtype
#def predict_view(week, user_set):
def predict_view():
    with open("user_dic.pkl", 'rb') as f:
        user_dic = pickle.load(f)

    user_set = user_dic.keys()
    train = []
    test = []
    y_train = [0]*3*len(user_set)
    y_test = [0]*len(user_set)
    #st = 3 # 1,2,3,4
    start =1
    ind = 0
    for i in range(len(user_set)):
        if ind % 10000 == 0:
            print ind
        uid = user_set[i]
        hist = user_dic[uid]
        #for t in range(10):
        #    for s in range(len(hist[t])):
        #        if hist[t][s] > 0:
        #            hist[t][s] = 1

        tr = []
        #te = []
        st = start
        for t in range(10):
            tr.extend(hist[t][st*7:(st+2)*7])
            if sum(hist[t][(st+2)*7 : (st+3)*7]) > 0:
                y_train[ind] = 1

            #te.extend(hist[t][(st+1)*7:(st+3)*7])
            #if sum(hist[t][(st+3)*7 : (st+4)*7]) > 0:
            #    y_test[ind] = 1
        train.append(tr)
        #test.append(te)

        ind += 1
        tr = []
        #te = []
        st = start + 1
        for t in range(10):
            tr.extend(hist[t][st*7:(st+2)*7])
            if sum(hist[t][(st+2)*7 : (st+3)*7]) > 0:
                y_train[ind] = 1

            #te.extend(hist[t][(st+1)*7:(st+3)*7])
            #if sum(hist[t][(st+3)*7 : (st+4)*7]) > 0:
            #    y_test[ind] = 1
        train.append(tr)
        #test.append(te)

        ind += 1
        tr = []
        te = []
        st = start + 2
        for t in range(10):
            tr.extend(hist[t][st*7:(st+2)*7])
            if sum(hist[t][(st+2)*7 : (st+3)*7]) > 0:
                y_train[ind] = 1

            te.extend(hist[t][(st+1)*7:(st+3)*7])
            if sum(hist[t][(st+3)*7 : (st+4)*7]) > 0:
                y_test[i] = 1
        train.append(tr)
        test.append(te)

        ind += 1

    #model = LogisticRegression()
    #model.fit(train, y_train)
    #pred = model.predict(test)

    params = {"objective": "binary:logistic",
              #"boost": "gblinear",
              #"eta": 0.02,
              #"max_depth": 10,
              #"subsample": 0.9,
              #"colsample_bytree": 1, #0.7,
              #"min_child_weight": 3,
              #"seed": 1301,
              "silent": 1,
              "nthread": 16
              }
    num_trees = 100

    dtrain = xgb.DMatrix(train, y_train)
    watchlist = [(dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=100)
    pred = gbm.predict(xgb.DMatrix(test))

    print len([1 for i in pred if i > 0.9])
    print len([1 for i in pred if i > 0.8])
    print len([1 for i in pred if i > 0.7])
    print len([1 for i in pred if i > 0.6])
    print len([1 for i in pred if i > 0.5])

    print roc_auc_score(y_test, pred)
    #return pred


if __name__ == '__main__':
    start = time.time()
    predict_view()

    print "Cost time is %s" %( (time.time() - start) / 1000 )
