import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

from utils import Gini


def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 5
    params["subsample"] = 0.85
    params["colsample_bytree"] = 0.85
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 8

    plst = list(params.items())

    #Using 5000 rows for early stopping.
    offset = 4000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
#    xgtrain = xgb.DMatrix(train, label=labels)
    #xgval = xgb.DMatrix(train, label=labels)

    #train using early stopping and predict
    watchlist = [(xgtrain, "train"),(xgval, "val")]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
    #model = xgb.train(plst, xgtrain, 1000)
    pred1_val = model.predict(xgval)
    print Gini(labels[:offset], pred1_val)
    preds1 = model.predict(xgtest)


    #reverse train and labels and use different 5k for early stopping.
    # this adds very little to the score but it is an option if you are concerned about using all the data.
    train = train[::-1,:]
    labels = np.log(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
#    xgtrain = xgb.DMatrix(train, label=labels)


    watchlist = [(xgtrain, "train"),(xgval, "val")]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
#    model = xgb.train(plst, xgtrain, 1000)
    pred2_val = model.predict(xgval)
    print Gini(labels[:offset], pred2_val)
    preds2 = model.predict(xgtest)


    #combine predictions
    #since the metric only cares about relative rank we don"t need to average
    preds = preds1*2.6 + preds2*7.4
    return preds

print 'xgboost, two time ensemble'

#load train and test
train  = pd.read_csv("data/train.csv", index_col=0)
test  = pd.read_csv("data/test.csv", index_col=0)


labels = train.Hazard
train.drop("Hazard", axis=1, inplace=True)

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

columns = train.columns
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


preds1 = xgboost_pred(train_s,labels,test_s)

#model_2 building

train = train.T.to_dict().values()
test = test.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

preds2 = xgboost_pred(train,labels,test)


preds = 0.6 * preds1 + 0.4 * preds2

flag = True

if flag:
    exit(0)

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index("Id")
preds.to_csv("xgboost_benchmark_kk.csv")
