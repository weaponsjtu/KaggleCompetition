###
# ensemble.py
# author: Weipeng Zhang
#
#
# 1. check each weight by hyperopt
# 2. apply the weight to train/test
###

import cPickle as pickle
import numpy as np
import pandas as pd
import sys

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, pyll

from param import config

from utils import *

def ensemble_selection_obj(param, model1_pred, model2_pred, labels, num_valid_matrix):
    weight = param['weight']
    gini_cv = np.zeros((config.kiter, config.kfold), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            p1 = model1_pred[iter, fold, :num_valid_matrix[iter, fold]]
            p2 = model2_pred[iter, fold, :num_valid_matrix[iter, fold]]
            y_pred = (p1 + p2*weight) / (1+weight)
            y_true = labels[iter, fold, :num_valid_matrix[iter, fold]]
            score = Gini(y_true, y_pred)
            gini_cv[iter][fold] = score
    gini_mean = np.mean(gini_cv)
    return -gini_mean


def ensemble_selection():
    # load feat, labels and pred
    feat_names = config.feat_names

    model_list = config.model_list

    # num valid matrix
    num_valid_matrix = np.zeros((config.kiter, config.kfold), dtype=int)

    # load valid labels
    valid_labels = np.zeros((config.kiter, config.kfold, 50000), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            path = "%s/iter%d/fold%d" % (config.data_folder, iter, fold)
            label_file = "%s/valid.%s.feat.pkl" %(path, feat_names[0])
            with open(label_file, 'rb') as f:
                [x_val, y_true] = pickle.load(f)
            valid_labels[iter, fold, :y_true.shape[0]] = y_true
            num_valid_matrix[iter][fold] = y_true.shape[0]
    maxNumValid = np.max(num_valid_matrix)

    # load all predictions, cross validation
    # compute model's gini cv score
    #gini_cv = np.zeros((len(feat_name), len(model_list), dtype=float)
    gini_cv = []
    model_valid_pred = np.zeros((len(feat_names)*len(model_list), config.kiter, config.kfold, maxNumValid), dtype=float)
    #model_valid_pred = []

    for feat in range(len(feat_names)):
        for mid in range(len(model_list)):
            gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
            for iter in range(config.kiter):
                for fold in range(config.kfold):
                    path = "%s/iter%d/fold%d" % (config.data_folder, iter, fold)
                    pred_file = "%s/%s_%s.pred.pkl" %(path, feat_names[feat], model_list[mid])
                    with open(pred_file, 'rb') as f:
                        y_pred = pickle.load(f)
                    model_valid_pred[feat*len(model_list) + mid][iter][fold][:num_valid_matrix[iter][fold]] = y_pred
                    #model_valid_pred.append(y_pred)
                    score = Gini(valid_labels[iter, fold, :num_valid_matrix[iter, fold]], y_pred)
                    gini_cv_tmp[iter][fold] = score
            gini_cv.append(np.mean(gini_cv_tmp))

    # sort the model by their cv mean score
    gini_cv = np.array(gini_cv)
    sorted_model = gini_cv.argsort()[::-1]
    # TODO

    # boosting ensemble
    # 1. initialization, use the max score model
    model_pred_tmp = np.zeros((config.kiter, config.kfold, maxNumValid), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            model_pred_tmp[iter, fold, :num_valid_matrix[iter][fold]] = model_valid_pred[sorted_model[0], iter, fold, :num_valid_matrix[iter][fold]]
    fid = sorted_model[0]/len(model_list)
    mid = sorted_model[0]%len(model_list)
    print "Init with best model, Gini %f, Model %s - %s" %(np.max(gini_cv), feat_names[fid], model_list[mid])

    # 2. greedy search
    best_model_list = []
    best_weight_list = []

    best_gini = np.max(gini_cv)
    best_weight = None
    best_model = None
    ensemble_iter = 0
    while True:
        ensemble_iter += 1
        for model in sorted_model:
            # jump for the first max model
            if ensemble_iter == 1 and model == sorted_model[0]:
                continue

            obj = lambda param: ensemble_selection_obj(param, model_pred_tmp, model_valid_pred[model], valid_labels, num_valid_matrix)
            param_space = {
                'weight': hp.quniform('weight', 0, 1, 0.1),
            }
            trials = Trials()
            best_param = fmin(obj,
                space = param_space,
                algo = tpe.suggest,
                max_evals = 100,
                trials = trials)
            best_w = best_param['weight']

            gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
            for iter in range(config.kiter):
                for fold in range(config.kfold):
                    p1 = model_pred_tmp[iter, fold, :num_valid_matrix[iter, fold]]
                    p2 = model_valid_pred[model, iter, fold, :num_valid_matrix[iter, fold]]
                    y_true = valid_labels[iter, fold, :num_valid_matrix[iter, fold]]
                    y_pred = (p1 + p2*best_w) / (1+best_w)
                    score = Gini(y_true, y_pred)
                    gini_cv_tmp[iter, fold] = score


            fid = model/len(model_list)
            mid = model % len(model_list)
            print "Iter %d, Gini %f, Model %s - %s, Weight %f" %(ensemble_iter, np.mean(gini_cv_tmp), feat_names[fid], model_list[mid], best_w)
            if np.mean(gini_cv_tmp) > best_gini:
                best_gini, best_model, best_weight = np.mean(gini_cv_tmp), model, best_w
        if best_model == None:
            break
        fid = best_model/len(model_list)
        mid = best_model % len(model_list)
        print "Iter %d, Gini %f, Model %s - %s, Weight %f" %(ensemble_iter, best_gini, feat_names[fid], model_list[mid], best_weight)
        best_weight_list.append(best_weight)
        best_model_list.append(best_model)

        # reset the valid pred
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                p1 = model_pred_tmp[iter, fold, :num_valid_matrix[iter, fold]]
                p2 = model_valid_pred[best_model, iter, fold, :num_valid_matrix[iter, fold]]
                model_pred_tmp[iter, fold, :num_valid_matrix[iter][fold]] = (p1 + p2*best_weight)/(1+best_weight)

        best_model = None

    # save best model list
    with open("%s/best_model_list" % config.data_folder, 'wb') as f:
        pickle.dump([sorted_model, best_model_list, best_weight_list], f, -1)

def ensemble_prediction():
    # load best model list
    with open("%s/best_model_list" % config.data_folder, 'rb') as f:
        [sorted_model, best_model_list, best_weight_list] = pickle.load(f)

    model_list = config.model_list
    feat_names = config.feat_names

    # prediction, generate submission file
    path = "%s/all" % config.data_folder
    fid = sorted_model[0]/len(model_list)
    mid = sorted_model[0]%len(model_list)
    with open("%s/%s_%s.pred.pkl" %(path, feat_names[fid], model_list[mid]), 'rb') as f:
        y_pred = pickle.load(f)

    for i in range(len(best_model_list)):
        model = best_model_list[i]
        weight = best_weight_list[i]
        fid = sorted_model[model]/len(model_list)
        mid = sorted_model[model]%len(model_list)
        with open("%s/%s_%s.pred.pkl" %(path, feat_names[fid], model_list[mid]), 'rb') as f:
            y_pred_tmp = pickle.load(f)
        y_pred = (y_pred + y_pred_tmp*weight)/(1+weight)

    test = pd.read_csv(config.origin_test_path, index_col=0)
    idx = test.index
    preds = pd.DataFrame({"Id": idx, "Hazard": y_pred})
    preds = preds.set_index("Id")
    preds.to_csv("sub/model_library.csv")

if __name__ == "__main__":
    flag = sys.argv[1]
    if flag == "ensemble":
        ensemble_selection()
    if flag == "submission":
        ensemble_prediction()
