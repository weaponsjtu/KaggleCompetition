import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import cPickle as pickle


from hyperopt import hp

from utils import *


def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005  # [0,1]
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 9
    # add by weipeng
    #params['max_delta_step'] = 5  # not exact solution
    #params['lambda'] = 0.9  # L2 regularization, penalty
    #params['alpha'] = 0.9 # L1
    #params['gamma'] = 0.3
    #params['lambda_bias'] = 0.7
    #params['num_round'] = 200 # n_estimators

    plst = list(params.items())

    #Using 12000 rows for early stopping.
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
    score1 = Gini(labels[:offset], pred1_val)
    preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)


    #reverse train and labels and use different 5k for early stopping.
    # this adds very little to the score but it is an option if you are concerned about using all the data.
    train = train[::-1,:]
    labels = np.log(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
    #xgtrain = xgb.DMatrix(train, label=labels)
    #xgval= xgb.DMatrix(train, label=labels)


    watchlist = [(xgtrain, "train"),(xgval, "val")]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
#    model = xgb.train(plst, xgtrain, 800)
    pred2_val = model.predict(xgval, ntree_limit=model.best_iteration)
    score2 = Gini(labels[:offset], pred2_val)
    preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)


    #combine predictions
    #since the metric only cares about relative rank we don"t need to average
    preds = (preds1)*1.5 + (preds2)*8.5
    #preds = preds1*preds2
    print 'Gini Score is ', (score1+score2)/2
    return preds

def main(flag):
    #load train and test
    train  = pd.read_csv("data/train.csv", index_col=0)
    test  = pd.read_csv("data/test.csv", index_col=0)
    test_ind = test.index


    labels = train.Hazard
    train.drop("Hazard", axis=1, inplace=True)

    train_s = train
    test_s = test


    train_s.drop("T2_V10", axis=1, inplace=True)
    train_s.drop("T2_V7", axis=1, inplace=True)
    train_s.drop("T1_V13", axis=1, inplace=True)
    train_s.drop("T1_V10", axis=1, inplace=True)

    #train_s.drop("T1_V17", axis=1, inplace=True)
    #train_s.drop("T1_V17", axis=1, inplace=True)
    #train_s.drop("T2_V8", axis=1, inplace=True)

    test_s.drop("T2_V10", axis=1, inplace=True)
    test_s.drop("T2_V7", axis=1, inplace=True)
    test_s.drop("T1_V13", axis=1, inplace=True)
    test_s.drop("T1_V10", axis=1, inplace=True)

    #test_s.drop("T1_V17", axis=1, inplace=True)
    #test_s.drop("T1_V17", axis=1, inplace=True)
    #test_s.drop("T2_V8", axis=1, inplace=True)


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

    train_s, test_s = add_features(train_s, test_s)
    preds1 = xgboost_pred(train_s,labels,test_s)

    #model_2 building

    train = train.T.to_dict().values()
    test = test.T.to_dict().values()

    vec = DictVectorizer()
    train = vec.fit_transform(train)
    test = vec.transform(test)


    train, test = add_features(train, test)
    preds2 = xgboost_pred(train,labels,test)

    #preds = 0.463 * (preds1**0.01) + 0.537 * (preds2**0.01)
    #preds = preds1*preds2
    preds = preds1*preds2 / (preds1 + preds2)

    if flag:
        exit(0)

    #generate solution
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index("Id")
    preds.to_csv("xgb_last.csv")

from sklearn.decomposition import PCA
def pca_wrapper(array):
    pca = PCA(n_components=2)
    array = pca.fit(array)
    return array

def add_features(train, test):
    if type(train) != np.ndarray:
        train = train.toarray()

    if type(test) != np.ndarray:
        test = test.toarray()

    with open('train_feature_engineered.pkl') as f:
        df_train = pickle.load(f)
    with open('test_feature_engineered.pkl') as f:
        df_test = pickle.load(f)

    feats = ['meanH_N_1', 'meanH_N_2', 'meanH_N_4', 'meanH_N_8']
    new_feature = np.zeros(( len(df_train.index), 4 ), dtype=float)
    for i in range(4):
        new_feature[:, i] = df_train[ feats[i] ].values
    train = np.append(train, new_feature, 1)

    new_feature = np.zeros(( len(df_test.index), 4 ), dtype=float)
    for i in range(4):
        new_feature[:, i] = df_test[ feats[i] ].values
    test = np.append(test, new_feature, 1)
    print "add new feature done!!!"

    return train, test


if __name__ == '__main__':
    print 'xgb model, pow'
    flag = False
    main(flag)
