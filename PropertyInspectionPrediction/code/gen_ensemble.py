###
# ensemble.py
# author: Weipeng Zhang
#
#
# 1. check each weight by hyperopt
# 2. apply the weight to train/test
###

from sklearn.metrics import mean_squared_error as MSE

import cPickle as pickle
import numpy as np
import pandas as pd
import sys,os

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, pyll
from hyperopt.mongoexp import MongoTrials

from param import config

from utils import *


def ensemble_algorithm(p1, p2, weight):
    #return (p1 + weight*p2) / (1+weight)
    return weight*p1 + (1-weight)*p2

def gen_subm(y_pred, filename=None):
    test = pd.read_csv(config.origin_test_path, index_col=0)
    idx = test.index
    preds = pd.DataFrame({"Id": idx, "Hazard": y_pred})
    preds = preds.set_index("Id")
    if filename != None:
        preds.to_csv(filename)
    else:
        preds.to_csv("sub/model_library.csv")

def add_prior_models(model_library):
    #prior_models = {
    #        'xgboost-art@1': {
    #            'weight': 0.463,
    #            'pow_weight': 0.01,
    #            },
    #        'xgboost-art@2': {
    #            'weight': 0.463,
    #            'pow_weight': 0.8,
    #            },
    #        'xgboost-art@3': {
    #            'weight': 0.463,
    #            'pow_weight': 0.045,
    #            'pow_weight1': 0.055,
    #            },
    #        'xgboost-art@4': {
    #            'weight': 0.463,
    #            'pow_weight': 0.98,
    #            },
    #        'xgboost-art@5': {
    #            'weight': 0.463,
    #            'pow_weight': 1,
    #            },
    #        'xgboost-art@6': {
    #            'weight': 0.47,
    #            'pow_weight': 1,
    #            },
    #        }
    prior_models = {}
    for i in range(1, 5):
        model = 'xgboost-art@%d'%i
        prior_models[model] = {'weight': 0.463, 'pow_weight': 0.01 * i}

    feat_names = config.feat_names
    model_list = config.model_list
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
            with open("%s/label_encode_xgboost-art.pred.pkl" %path, 'rb') as f:
                p1 = pickle.load(f)
            with open("%s/dictvec_xgboost-art.pred.pkl" %path, 'rb') as f:
                p2 = pickle.load(f)
            for model in prior_models.keys():
                weight = prior_models[model]['weight']
                pow_weight1 = prior_models[model]['pow_weight']
                pow_weight2 = pow_weight1
                if prior_models[model].has_key('pow_weight1'):
                    pow_weight2 = prior_models[model]['pow_weight1']
                pred = weight * (p1**pow_weight1) + (1-weight) * (p2**pow_weight2)
                with open("%s/%s.pred.pkl" %(path, model), 'wb') as f:
                    pickle.dump(pred, f, -1)

    path = "%s/all" %(config.data_folder)
    with open("%s/label_encode_xgboost-art.pred.pkl" %path, 'rb') as f:
        p1 = pickle.load(f)
    with open("%s/dictvec_xgboost-art.pred.pkl" %path, 'rb') as f:
        p2 = pickle.load(f)
    for model in prior_models.keys():
        weight = prior_models[model]['weight']
        pow_weight1 = prior_models[model]['pow_weight']
        pow_weight2 = pow_weight1
        if prior_models[model].has_key('pow_weight1'):
            pow_weight2 = prior_models[model]['pow_weight1']
        pred = weight * (p1**pow_weight1) + (1-weight) * (p2**pow_weight2)
        with open("%s/%s.pred.pkl" %(path, model), 'wb') as f:
            pickle.dump(pred, f, -1)

    for model in prior_models.keys():
        model_library.append(model)
    return model_library



def ensemble_selection_obj(param, model1_pred, model2_pred, labels, num_valid_matrix):
    weight = param['weight']
    gini_cv = np.zeros((config.kiter, config.kfold), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            p1 = model1_pred[iter, fold, :num_valid_matrix[iter, fold]]
            p2 = model2_pred[iter, fold, :num_valid_matrix[iter, fold]]
            y_pred = ensemble_algorithm(p1, p2, weight)

            y_true = labels[iter, fold, :num_valid_matrix[iter, fold]]
            score = Gini(y_true, y_pred)
            gini_cv[iter][fold] = score
    gini_mean = np.mean(gini_cv)
    return -gini_mean

def check_model(model_name):
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            if os.path.exists('%s/iter%d/fold%d/%s.pred.pkl' %(config.data_folder, iter, fold, model_name)) is False:
                return False

    if os.path.exists('%s/all/%s.pred.pkl' %(config.data_folder, model_name)) is False:
        return False

    return True


def ensemble_selection():
    # load feat, labels and pred
    feat_names = config.feat_names
    model_list = config.model_list

    # combine them, and generate whold model_list
    model_library = []
    for feat in feat_names:
        for model in model_list:
            if check_model("%s_%s"%(feat, model)):
                model_library.append("%s_%s" %(feat, model))
            for num in range(1, config.hyper_max_evals+1):
                model_name = "%s_%s@%d" %(feat, model, num)
                if check_model(model_name):
                    model_library.append(model_name)

    #model_library = add_prior_models(model_library)
    print model_library
    model_num = len(model_library)

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
    gini_cv = []
    model_valid_pred = np.zeros((model_num, config.kiter, config.kfold, maxNumValid), dtype=float)

    for mid in range(model_num):
        gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                path = "%s/iter%d/fold%d" % (config.data_folder, iter, fold)
                pred_file = "%s/%s.pred.pkl" %(path, model_library[mid])
                with open(pred_file, 'rb') as f:
                    y_pred = pickle.load(f)
                model_valid_pred[mid, iter, fold, :num_valid_matrix[iter, fold]] = y_pred
                score = Gini(valid_labels[iter, fold, :num_valid_matrix[iter, fold]], y_pred)
                gini_cv_tmp[iter][fold] = score
        gini_cv.append(np.mean(gini_cv_tmp))

    # sort the model by their cv mean score
    gini_cv = np.array(gini_cv)
    sorted_model = gini_cv.argsort()[::-1]

    # boosting ensemble
    # 1. initialization, use the max score model
    model_pred_tmp = np.zeros((config.kiter, config.kfold, maxNumValid), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            model_pred_tmp[iter, fold, :num_valid_matrix[iter][fold]] = model_valid_pred[sorted_model[0], iter, fold, :num_valid_matrix[iter][fold]]
    print "Init with best model, Gini %f, Model %s" %(np.max(gini_cv), model_library[sorted_model[0]])

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
            print "ensemble iter %d, model %d" %(ensemble_iter, model)
            # jump for the first max model
            if ensemble_iter == 1 and model == sorted_model[0]:
                continue

            obj = lambda param: ensemble_selection_obj(param, model_pred_tmp, model_valid_pred[model], valid_labels, num_valid_matrix)
            param_space = {
                'weight': hp.quniform('weight', 0, 1, 0.001),
            }
            trials = Trials()
            #trials = MongoTrials('mongo://172.16.13.7:27017/ensemble/jobs', exp_key='exp%d_%d'%(ensemble_iter, model))
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
                    y_pred = ensemble_algorithm(p1, p2, best_w)
                    score = Gini(y_true, y_pred)
                    gini_cv_tmp[iter, fold] = score


            print "Iter %d, Gini %f, Model %s, Weight %f" %(ensemble_iter, np.mean(gini_cv_tmp), model_library[model], best_w)
            if np.mean(gini_cv_tmp) > best_gini:
                best_gini, best_model, best_weight = np.mean(gini_cv_tmp), model, best_w
        if best_model == None:
            break
        print "Best for Iter %d, Gini %f, Model %s, Weight %f" %(ensemble_iter, best_gini, model_library[best_model], best_weight)
        best_weight_list.append(best_weight)
        best_model_list.append(best_model)

        # reset the valid pred
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                p1 = model_pred_tmp[iter, fold, :num_valid_matrix[iter, fold]]
                p2 = model_valid_pred[best_model, iter, fold, :num_valid_matrix[iter, fold]]
                model_pred_tmp[iter, fold, :num_valid_matrix[iter][fold]] = (1-best_weight)*p1 + best_weight*p2

        best_model = None
        print 'ensemble iter %d done!!!' % ensemble_iter

    # save best model list
    with open("%s/best_model_list" % config.data_folder, 'wb') as f:
        pickle.dump([model_library, sorted_model, best_model_list, best_weight_list], f, -1)

def ensemble_prediction():
    # load best model list
    with open("%s/best_model_list" % config.data_folder, 'rb') as f:
        [model_library, sorted_model, best_model_list, best_weight_list] = pickle.load(f)

    # prediction, generate submission file
    path = "%s/all" % config.data_folder
    print "Init with (%s)" %(model_library[sorted_model[0]])
    with open("%s/%s.pred.pkl" %(path, model_library[sorted_model[0]]), 'rb') as f:
        y_pred = pickle.load(f)

    for i in range(len(best_model_list)):
        model = best_model_list[i]
        weight = best_weight_list[i]
        print "(%s), %f" %(model_library[model], weight)
        with open("%s/%s.pred.pkl" %(path, model_library[model]), 'rb') as f:
            y_pred_tmp = pickle.load(f)
        y_pred = ensemble_algorithm(y_pred, y_pred_tmp, weight)

    gen_subm(y_pred)


if __name__ == "__main__":
    flag = sys.argv[1]
    print "start ", flag
    if flag == "ensemble":
        ensemble_selection()
    if flag == "submission":
        ensemble_prediction()
