
class ParamConfig:
    def __init__(self):
        self.kfold = 2  # cross validation, k-fold
        self.kiter = 1  # shuffle dataset, and repeat CV
        self.model_list = ['linear', 'logistic', 'svm', 'randomforest', 'xgboost', 'nn']
        self.model_type = 'RankSVM'
