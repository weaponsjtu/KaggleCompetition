
from hyperopt import hp, pyll

class ParamConfig:
    def __init__(self):
        self.kfold = 2  # cross validation, k-fold
        self.kiter = 1  # shuffle dataset, and repeat CV
        self.model_list = ['linear', 'logistic', 'svr', 'ranksvm', 'rf', 'extratree', 'xgboost', 'knn']
        self.model_type = 'xgboost'
        self.param_spaces = {
            'linear': {
            },
            'logistic': {
            },
            'knn': {
                'n_neighbors': pyll.scope.int(hp.quniform('n_neighbors', 5, 6, 1)),
                'weights': hp.choice('weights', ['uniform', 'distance']),
            },
            'rf': {
                'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 1000, 100)),
            },
            'extratree': {
                'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 1000, 100)),
            },
            'svr': {
                'C': hp.quniform('C', 0.1, 10, 0.1),
            },
            'xgboost': {
                'booster': 'gbtree',
                'objective': 'reg:linear',
                'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 1000, 100)),
                'eta': hp.quniform('eta', 0.01, 1, 0.01),
                'gamma': hp.quniform('gamma', 0, 2, 0.1),
                'min_child_weight': pyll.scope.int( hp.quniform('min_child_weight', 0, 10, 1) ),
                'subsample': hp.quniform('subsample', 0.7, 0.9, 0.05),
                'silent': 0,
                'verbose': 0,
                'max_depth': pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                'num_rounds': 10000,
            },
        }
        self.best_param = {
            'linear': {
            },
            'logistic': {
            },
            'knn': {
                'n_neighbors': 8,
                'weights': 'distance',
            },
            'rf': {
                'n_estimators': 100,
            },
            'extratree': {
                'n_estimators': 100,
            },
            'svr': {
                'C': 1,
            },
            'xgboost': {
                'booster': 'gbtree',
                'objective': 'reg:linear',
                'eta': 0.005,
                'min_child_weight': 5,
                'subsample': 0.75,
                'silent': 0,
                'max_depth': 8,
                'num_rounds': 10000,
            },
        }
