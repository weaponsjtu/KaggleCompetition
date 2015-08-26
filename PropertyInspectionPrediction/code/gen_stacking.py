###
# gen_stacking.py
# author: Weipeng Zhang
#
#
# 1. prediction feature, LogisticRegression
# 2. prediction feature + origin feature, non-linear, GBF, KNN, ET, etc
###

from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

import cPickle as pickle
import numpy as np
import pandas as pd
import sys,os,time
import multiprocessing

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, pyll
from hyperopt.mongoexp import MongoTrials

from param import config

from utils import *

from param import config
from gen_ensemble import gen_model_library, gen_subm

# stacking
def model_stacking():
    # load data
    train = pd.read_csv(config.origin_train_path, index_col=0)
    test = pd.read_csv(config.origin_test_path, index_col=0)

    x_label = train['Hazard'].values
    y_len = len(list(test.index))


    model_library = gen_model_library()
    if model_library.count('label_xgb_art@24') > 0:
        print 'label_xgb_art@24'
    print len(model_library)
    model_library = [ 'label_xgb_art@24' ]
    print model_library
    blend_train = np.zeros((len(x_label), len(model_library), config.kiter))
    blend_test = np.zeros((y_len, len(model_library)))

    # load kfold object
    with open("%s/fold.pkl" % config.data_folder, 'rb') as i_f:
        skf = pickle.load(i_f)
    i_f.close()

    for iter in range(config.kiter):
        for i in range(len(model_library)):
            for j, (validInd, trainInd) in enumerate(skf[iter]):
                path = "%s/iter%d/fold%d/%s.pred.pkl" %(config.data_folder, iter, j, model_library[i])
                with open(path, 'rb') as f:
                    y_pred = pickle.load(f)
                f.close()
                blend_train[validInd, i, iter] = y_pred


    for i in range(len(model_library)):
        path = "%s/all/%s.pred.pkl" %(config.data_folder, model_library[i])
        with open(path, 'rb') as f:
            y_pred = pickle.load(f)
        f.close()
        blend_test[:, i] = y_pred

    print "Blending..."
    clf = RandomForestRegressor()
    y_sub = np.zeros((y_len))
    for iter in range(config.kiter):
        clf.fit(blend_train[:, :, iter], x_label)
        y_pred = clf.predict(blend_test)
        y_sub += y_pred
    y_sub = y_sub / config.kiter
    gen_subm(y_sub, 'sub/model_stack.csv')

if __name__ == '__main__':
    model_stacking()
