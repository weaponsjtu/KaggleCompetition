import numpy as np
import pandas as pd
import cPickle as pickle
import os

from param import config

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) *1.0 / np.sum(true_order)
    L_pred = np.cumsum(pred_order) *1.0/ np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    #print "pred gini is %f, true gini is %f" %(G_pred, G_true)
    return G_pred/G_true

def stretch_lr(y_pred):
    y_pred = np.array(y_pred)
    #y_pred = (y_pred - y_pred.min()) * 1.0 / (y_pred.max() - y_pred.min())
    y_pred = y_pred * 1.0 / y_pred.max()
    return y_pred

def cv_split(train_z, labels_z, kfold, kiter):
    train_subsets_k = []
    label_subsets_k = []
    n_sample = train_z.shape[0] / kfold
    data = np.column_stack((train_z, labels_z))
    for k in range(kiter):
        np.random.shuffle(data)
        train = data[:, :-1]
        labels = data[:, -1]
        train_subsets = []
        label_subsets = []
        for i in range(kfold):
            tmp_train = np.array( np.concatenate( (np.copy(train[0:(i*n_sample), :]), np.copy(train[((i+1)*n_sample):, :])), axis=0 ), copy=True )
            tmp_val = np.array(train[(i * n_sample):((i+1)*n_sample), :], copy=True)

            tmp_train_label = np.array( np.concatenate( (np.copy(labels[0:(i*n_sample)]), np.copy(labels[((i+1)*n_sample):])), axis=0 ), copy=True)
            tmp_val_label = np.array(labels[(i*n_sample):((i+1)*n_sample)], copy=True)

            #print 'train', i, tmp_train[1,:10]
            #print 'val', i, tmp_val[1,:10]
            train_subsets.append([ tmp_train, tmp_val ])
            label_subsets.append([ tmp_train_label, tmp_val_label ])
        train_subsets_k.append( train_subsets )
        label_subsets_k.append( label_subsets )
    return train_subsets_k, label_subsets_k


def write_submission(idx, pred, filename):
    preds = pd.DataFrame({"Id": idx, "Hazard": pred})
    preds = preds.set_index("Id")
    preds.to_csv(filename)

def show_best_params():
    with open("%s/model_best_params" %config.data_folder, 'rb') as f:
        model_best_params = pickle.load(f)
    f.close()
    print model_best_params.keys()

    #model_best_params.pop('label_xgb_art')
    #model_best_params.pop('dictvec_xgb_art')
    #print model_best_params.keys()
    #with open("%s/model_best_params" %config.data_folder, 'wb') as f:
    #    pickle.dump(model_best_params, f, -1)
    #f.close()


def append_best_params():
    model_best_params = {}
    if os.path.exists("%s/model_best_params"%config.data_folder):
        with open("%s/model_best_params" %config.data_folder, 'rb') as f:
            model_best_params = pickle.load(f)
        f.close()

    model_best_params['standard_logistic'] = {
            'C': 0.002656192975371555}
    model_best_params['standard_knn'] = {
            'n_neighbors': 150}
    model_best_params['standard_ridge'] = {
            'alpha': 19.865153638453595}
    model_best_params['standard_lasso'] = {
            'alpha': 0.09918358854815801}
    model_best_params['standard_xgb_rank'] = {
            'lambda_bias': 0.2,
            'alpha': 0.24,
            'eta': 0.06,
            'lambda': 5.0}
    model_best_params['standard_xgb_linear'] = {
            'lambda_bias': 1.8,
            'alpha': 0.22,
            'eta': 0.01,
            'lambda': 3.15}

    model_best_params['standard_xgb_tree'] = {
            'subsample': 0.9,
            'eta': 0.01,
            'max_depth': 2.0,
            'gamma': 0.8,
            'min_child_weight': 2.0}

    with open("%s/model_best_params" %config.data_folder, 'wb') as f:
        pickle.dump(model_best_params, f, -1)
    f.close()

def change_name():
    feat_names = config.feat_names
    model_list = config.model_list
    # cross validation
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
            for feat in feat_names:
                for model in model_list:
                    for num in range(1, config.hyper_max_evals + 1):
                        pred_file = "%s/%s_%s.pred.%d.pkl" %(path, feat, model, num)
                        if os.path.exists(pred_file):
                            os.rename(pred_file, "%s/%s_%s@%d.pred.pkl" %(path, feat, model, num))
    # train/test
    path = "%s/all" %(config.data_folder)
    for feat in feat_names:
        for model in model_list:
            for num in range(1, config.hyper_max_evals + 1):
                pred_file = "%s/%s_%s.pred.%d.pkl" %(path, feat, model, num)
                if os.path.exists(pred_file):
                    os.rename(pred_file, "%s/%s_%s@%d.pred.pkl" %(path, feat, model, num))

if __name__ == '__main__':
    show_best_params()
    #change_name()
    #append_best_params()
