import pandas as pd
import numpy as np
import cPickle as pickle

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

from param import config

def extract_feature(path, train, test, type, feat_names):
    y_train = train['Hazard'].values
    train.drop("Hazard", axis=1, inplace=True)
    y_test = [1] * len(test.index)
    if type == "valid":
        y_test = test['Hazard'].values
        test.drop("Hazard", axis=1, inplace=True)


    if feat_names.count("label_encode") > 0:
        train_s = train.copy()
        test_s = test.copy()
        train_s.drop("T2_V10", axis=1, inplace=True)
        train_s.drop("T2_V7", axis=1, inplace=True)
        train_s.drop("T1_V13", axis=1, inplace=True)
        train_s.drop("T1_V10", axis=1, inplace=True)

        test_s.drop("T2_V10", axis=1, inplace=True)
        test_s.drop("T2_V7", axis=1, inplace=True)
        test_s.drop("T1_V13", axis=1, inplace=True)
        test_s.drop("T1_V10", axis=1, inplace=True)

        train_s = np.array(train_s)
        test_s = np.array(test_s)

        #label encode the categorical variables
        for i in range(train_s.shape[1]):
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_s[:, i]) + list(test_s[:, i]))
            train_s[:, i] = lbl.transform(train_s[:, i])
            test_s[:, i] = lbl.transform(test_s[:, i])

        train_s = train_s.astype(float)
        test_s = test_s.astype(float)
        with open("%s/train.%s.feat.pkl" %(path, "label_encode"), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, "label_encode"), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)

    if feat_names.count("dictvec") > 0:
        train_s = train.copy()
        test_s = test.copy()
        train_s = train_s.T.to_dict().values()
        test_s = test_s.T.to_dict().values()

        vec = DictVectorizer()
        train_s = vec.fit_transform(train_s)
        test_s = vec.transform(test_s)
        with open("%s/train.%s.feat.pkl" %(path, "dictvec"), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, "dictvec"), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)



if __name__ == "__main__":
    # load data
    train = pd.read_csv(config.origin_train_path, index_col=0)
    test = pd.read_csv(config.origin_test_path, index_col=0)

    # load kfold object
    with open("%s/fold.pkl" % config.data_folder, 'rb') as i_f:
        skf = pickle.load(i_f)

    # extract features
    #feat_names = ["label_encode", "dictvec"]
    feat_names = config.feat_names

    # for cross validation
    #print "Extract feature for cross validation"
    #for iter in range(config.kiter):
    #    for fold, (validInd, trainInd) in enumerate(skf[iter]):
    #        path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
    #        sub_train = train.iloc[trainInd].copy()
    #        sub_val = train.iloc[validInd].copy()
    #        # extract feature
    #        extract_feature(path, sub_train, sub_val, "valid", feat_names)
    #print "Done"

    # for train/test
    print "Extract feature for train/test"
    path = "%s/all" % config.data_folder
    extract_feature(path, train, test, "test", feat_names)
    print "Done"
