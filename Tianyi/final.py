import pandas as pd
import numpy as np
from sklearn import cross_validation, preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import cPickle as pickle

from binary import predict_view

from scipy import stats
#import matplotlib.pyplot as plt
import statsmodels.api as sm
#import statsmodels.graphics.api as qqplot
import time, math


def ml_metric(mat_true, mat_pred):
    row = len(mat_true)
    col = len(mat_true[0])
    print "row & col is %d\t%d" %(row, col)
    print "row & col is %d\t%d" %(len(mat_pred), len(mat_pred[0]))
    similarity = 0.0
    comm_user = 0
    true_user = 0
    all_user = 0
    for i in range(row):
        y_true = mat_true[i]
        y_pred = mat_pred[i]
        if sum(y_true) > 0 and sum(y_pred) > 0:
            if i <= 3:
                print y_true
                print y_pred
            tmp_sum = 0.0
            sum_t = 0.0
            sum_p = 0.0
            for j in range(col):
                sum_t += y_true[j] * y_true[j]
                sum_p += y_pred[j] * y_pred[j]
                tmp_sum += y_true[j] * y_pred[j]
            tmp_sim = 0
            if sum_t > 0 and sum_p > 0:
                tmp_sim = tmp_sum / math.sqrt(sum_t * sum_p)

            similarity += tmp_sim

            # common user
            comm_user += 1

        # true, history user
        if sum(y_true) > 0:
            true_user += 1

        # predict user
        if sum(y_pred) > 0:
            all_user += 1

    print "all user is %d, true user is %d, comm user is %d" %(all_user, true_user, comm_user)

    precision = similarity / all_user
    recall = comm_user * 1.0 / true_user
    print "precision is %f, recall is %f" %(precision, recall)
    return 2 * precision * recall / (precision + recall)

####
# 1. history count, 1 week, 2 week, 3 week
# 2. has not viewed
####
def build_features(train, test=None):
    # userid, date, type, week, day, count
    print train.columns
    user_set = np.unique(train['userid'])
    y_train = train['count']

    train['userid'] = train['userid'].apply(lambda x: user_set.searchsorted(x))

    with open("user_dic.pkl", 'rb') as f:
        user_dic = pickle.load(f)

    # add new features
    print "Add new features"
    type_num = 10

    has_not_view = [0] * len(train.index)
    history_day_q = [0] * len(train.index)
    history_q = [0] * len(train.index)
    history_one_week_day_q = [0] * len(train.index)
    history_one_week_q = [0] * len(train.index)
    history_two_week_day_q = [0] * len(train.index)
    history_two_week_q = [0] * len(train.index)
    history_type_q = [[0]*type_num] * len(train.index)
    history_type_day_q = [[0]*type_num] * len(train.index)
    for i in range(len(train.index)):
        if i%10000==0:
            print i
        u = train['userid'].iloc[i]
        w = train['week'].iloc[i]
        d = train['day'].iloc[i]
        t = train['type'].iloc[i]

        uid = user_set[u]
        hist = user_dic[uid][t-1]
        if sum(hist[:(w-1)*7]) > 0:
            has_not_view[i] = 1
            history_q[i] = sum(hist[:(w-1)*7]) * 1.0 / (w-1)*7


        tmp = hist[:(w-1)*7]
        for c in range(d-1, len(tmp), 7):
            history_day_q[i] += tmp[c]

        if w >= 2:
            tmp = hist[(w-2)*7 : (w-1)*7]
            history_one_week_q[i] = sum(tmp)
            history_one_week_day_q[i] = tmp[d-1]

        ###### query
        #
        #tmp = train.query('userid==@u')
        #tmp = tmp.query('type == @t & week < @w')

        #if sum(tmp['count']) > 0:
        #    has_not_view[i] = 1

        #tmp_day = tmp.query('day == @d')
        #history_day_q[i] = sum(tmp_day['count'])

        #w = w - 1
        #tmp_week = tmp.query('week == @w & day == @d')
        #history_one_week_day_q = sum(tmp_week['count'])

        #tmp_week  = tmp.query('week == @w')
        #history_one_week_q = sum(tmp_week['count'])
        #######

    train['history_day_q'] = history_day_q
    train['has_not_view'] = has_not_view
    train['history_one_week_day_q'] = history_one_week_day_q
    train['history_one_week_q'] = history_one_week_q

    print "generate test data"
    target_week = 8
    if test is not None:
        target_week = 7

    pred_num = len(user_set) * 7 * 10

    userid_set = [0] * pred_num
    type_set = [0] * pred_num
    week_set = [target_week] * pred_num
    day_set = [0] * pred_num
    history_day_q = [0] * pred_num
    has_not_view = [0] * pred_num
    history_one_week_q = [0] * pred_num
    history_one_week_day_q = [0] * pred_num

    w = target_week
    for i in range(len(user_set)):
        uid = user_set[i]
        for d in range(1,8):
            for t in range(1,11):
                index = i * 70 + (d-1) * 7 + t - 1
                if index%10000==0:
                    print index

                userid_set[index] = i
                type_set[index] = t
                day_set[index] = d

                hist = user_dic[uid][t-1]
                if sum(hist[:(w-1)*7]) > 0:
                    has_not_view[index] = 1

                tmp = hist[:(w-1)*7]
                for c in range(d-1, len(tmp), 7):
                    history_day_q[index] += tmp[c]

                tmp = hist[(w-2)*7 : (w-1)*7]
                history_one_week_q[index] = sum(tmp)
                history_one_week_day_q[index] = tmp[d-1]

                ##### query
                #tmp = train.query('userid==@i')
                #tmp = tmp.query('type==@t')
                #pred['userid'].iloc[index] = i
                #pred['type'].iloc[index] = t
                #pred['day'].iloc[index] = d

                #if sum(tmp['count']) > 0:
                #    pred['has_not_view'].iloc[index] = 1

                #tmp_day = tmp.query('day == @d')
                #pred['history_day_q'].iloc[index] = sum(tmp_day['count'])
                #w = target_week - 1
                #tmp_day = tmp.query('week == @w & day == @d')
                #pred['history_one_week_day_q'].iloc[index] = sum(tmp_day['count'])
                #tmp_day = tmp.query('week == @w')
                #pred['history_one_week_q'].iloc[index] = sum(tmp_day['count'])
                ######

    pred = pd.DataFrame({'userid': userid_set, 'type': type_set, 'week': week_set, 'day': day_set, 'history_day_q': history_day_q, 'has_not_view': has_not_view, 'history_one_week_q': history_one_week_q, 'history_one_week_day_q': history_one_week_day_q})



    label = np.zeros([len(user_set), 7*10])
    if test is not None:
        print "generate label"
        for i in range(len(test.index)):
            u = test['userid'].iloc[i]
            d = test['day'].iloc[i]
            t = test['type'].iloc[i]
            index = user_set.searchsorted(u)
            if index < len(user_set) and user_set[index] == u:
                label[index][(d-1)*10 + t - 1] = test['count'].iloc[i]
        with open('feat/label.pkl', 'wb') as f:
            pickle.dump(label, f, -1)

    train.drop(['date', 'count'], axis=1, inplace=True)

    print "dump data"
    filename = "feat/feature.pkl"
    if test is not None:
        filename = "feat/feature_cross.pkl"

    with open(filename, 'wb') as f:
        pickle.dump([train, y_train, pred, label, user_set], f, -1)

def preprocess(flag):
    print "preprocess data"
    data = pd.read_csv('data/train.csv', sep='\t')

    features = ['userid', 'type', 'week', 'day']

    data['week'] = data.date.apply(lambda x: x.split('w')[1][:1])
    data['week'] = data['week'].astype(int)
    data['day'] = data.date.apply(lambda x: x.split('d')[1])
    data['day'] = data['day'].astype(int)
    data['type'] = data.type.apply(lambda x: x.split('v')[1])
    data['type'] = data['type'].astype(int)

    if flag == "submission":
        build_features(data)
    else:
        train = data[ data['week'] < 7 ].copy()
        test = data[ data['week'] >= 7 ].copy()
        build_features(train, test)

def train_model(flag):
    filename = "feat/feature.pkl"
    if flag == "cross":
        filename = "feat/feature_cross.pkl"
    with open(filename, 'rb') as f:
        [train, y_train, pred, label, user_set] = pickle.load(f)

    #features = ['userid', 'type', 'week', 'day', 'history_day_q', 'has_not_view', 'history_one_week_q', 'history_one_week_day_q']
    features = ['userid', 'type', 'week', 'day', 'history_one_week_day_q', 'has_not_view']
    train = np.array(train[features])
    pred = np.array(pred[features])

    week = 8
    if flag == "cross":
        week = 7

    import time
    start_time = time.time()

    #######  different model ##############
    model_name = "super"
    #pred1 = linear_model(train, y_train, pred)
    #pred2 = xgb_model(train, y_train, pred)
    #test_probs = [math.sqrt(a*b) for a,b in zip(pred1, pred2)]
    #test_probs = [(a+b)/2 for a,b in zip(pred1, pred2)]
    #test_probs = [min(a, b) for a,b in zip(pred1, pred2)]

    #model_name = "lasagne"
    #test_probs = lasagne_model(train, y_train, pred)

    #model_name = "knn"
    #test_probs = knn_model(train, y_train, pred)

    #test_probs = xgb_model(train, y_train, pred)

    prob = predict_view(week, user_set)

    test_probs = np.array([0] * 70 * len(user_set))
    for i in range(len(user_set)):
        if prob[i] > 0.6:
            test_probs[i*70 : (i+1)*70] = [1] * 70

    ########################################

    #test_probs = np.rint(test_probs)
    test_probs = np.array(test_probs)

    if flag == "submission":
        output = open("sub/sub_"+model_name+".txt", 'wb')
        for i in range(len(user_set)):
            if sum(test_probs[i*70 : (i+1)*70]) > 0:
                res = str(user_set[i]) + '\t'
                for p in range(i * 70, (i+1) * 70):
                    res += str(int(test_probs[p])) + ','
                res = res[:-1]
                output.write(res + "\n")
        output.close()
    else:
        pred_mat = np.zeros([len(user_set), 70])
        for i in range(len(user_set)):
            pred_mat[i] = test_probs[i*70 : (i+1)*70].copy()
            #if i%10000 == 0:
            #    score = ml_metric(label, pred_mat)
            #    print "Index is %d, Score is %f"%(i, score)
        score = ml_metric(label, pred_mat)
        print "Score is ", score

    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )

def linear_model(train, y_train, test):
    model = LinearRegression()
    #model = LogisticRegression()
    #y_train = y_train * 1.0 / max_count
    #max_count = max(y_train)
    model.fit( train, y_train )
    test_probs = model.predict( test )

    indices = test_probs < 0
    test_probs[indices] = 0
    return test_probs

def xgb_model(train, y_train, test):
    print "start training with xgboost"
    params = {"objective": "reg:linear",
              #"boost": "gblinear",
              "eta": 0.02,
              "max_depth": 10,
              "subsample": 0.9,
              "colsample_bytree": 1, #0.7,
              #"min_child_weight": 3,
              #"seed": 1301,
              "silent": 1,
              "nthread": 16
              }
    num_trees = 100

    dtrain = xgb.DMatrix(train, y_train)
    watchlist = [(dtrain, 'train')]

    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=100)

    test_probs = gbm.predict(xgb.DMatrix(test))
    indices = test_probs < 0
    test_probs[indices] = 0
    return test_probs


def knn_model(train, y_train, test):
    model = KNeighborsRegressor(n_neighbors = 10, weights='distance', n_jobs=-1)
    model.fit(train, y_train)
    test_probs = model.predict(test)
    indices = test_probs < 0
    test_probs[indices] = 0
    return test_probs


from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax, sigmoid, rectify, tanh
from lasagne.objectives import categorical_crossentropy, binary_crossentropy, squared_error
#def lasagne_model(num_features, num_classes):
def lasagne_model(train, y_train, test):
    layers = [('input', InputLayer),
            ('dense0', DenseLayer),
            ('dropout0', DropoutLayer),
            ('dense1', DenseLayer),
            ('dropout1', DropoutLayer),
            ('dense2', DenseLayer),
            ('dropout2', DropoutLayer),
            ('output', DenseLayer)]

    num_features = len(train[0])
    num_classes = 1

    model = NeuralNet(layers=layers,
            input_shape=(None, num_features),
            objective_loss_function=squared_error,
            dense0_num_units=6,
            dropout0_p=0.4, #0.1,
            dense1_num_units=4,
            dropout1_p=0.4, #0.1,
            dense2_num_units=2,
            dropout2_p=0.4, #0.1,
            output_num_units=num_classes,
            output_nonlinearity=tanh,
            regression=True,
            update=nesterov_momentum, #adagrad,
            update_momentum=0.9,
            update_learning_rate=0.004,
            eval_size=0.2,
            verbose=1,
            max_epochs=5) #15)

    x_train = np.array(train).astype(np.float32)
    x_test = np.array(test).astype(np.float32)

    model.fit(x_train, y_train)
    pred_val = model.predict(x_test)
    print pred_val.shape
    test_probs = np.array(pred_val).reshape(len(pred_val),)
    print test_probs.shape

    indices = test_probs < 0
    test_probs[indices] = 0
    return test_probs

def fixed():
    pred = open('sub_xgb_org.txt', 'rb')
    output = open('sub_xgb_fix.txt', 'wb')
    for line in pred:
        temp = line.split(',')
        u = temp[0][:-1]
        res = u + '\t' + temp[0][-1] + ',' + ','.join(temp[1:])
        output.write(res)
    output.close()

import multiprocessing
class ModelProcess(multiprocessing.Process):
    def __init__(self, lock, user_dic, data, user_set):
        multiprocessing.Process.__init__(self)
        self.lock = lock
        self.user_dic = user_dic
        self.data = data
        self.user_set = user_set

    def run(self):
        for u in self.user_set:
            tmp = self.data.query('userid == @u').copy()
            self.user_dic.append(tmp)


def extract_user():
    data = pd.read_csv('data/train.csv', sep='\t')

    data['week'] = data.date.apply(lambda x: x.split('w')[1][:1])
    data['week'] = data['week'].astype(int)
    data['day'] = data.date.apply(lambda x: x.split('d')[1])
    data['day'] = data['day'].astype(int)
    data['type'] = data.type.apply(lambda x: x.split('v')[1])
    data['type'] = data['type'].astype(int)

    user_set = np.unique(data['userid'])
    #data['userid'] = data['userid'].apply(lambda x: user_set.searchsorted(x))

    step = len(user_set) / multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    mp_list = []
    user_dic_list = {}
    for i in range(multiprocessing.cpu_count()):
        start = i * step
        end = (i+1)*step if (i+1)*step < len(user_set) else len(user_set)
        mp_data = data.copy()
        user_set_part = user_set[start:end].copy()
        user_dic_list[i] = manager.list()
        mp= ModelProcess(lock, user_dic_list[i], mp_data, user_set)
        mp_list.append(mp)

    for mp in mp_list:
        mp.start()

    for mp in mp_list:
        mp.join()

    final_dic = []
    for i in range(multiprocessing.cpu_count()):
        dic_part = user_dic_list[i]
        final_dic.extend(dic_part)

    with open('user_dictionary.pkl', 'wb') as f:
        pickle.dump(final_dic, f, -1)

def extract_data():
    data = pd.read_csv('data/train.csv', sep='\t')

    data['week'] = data.date.apply(lambda x: x.split('w')[1][:1])
    data['week'] = data['week'].astype(int)
    data['day'] = data.date.apply(lambda x: x.split('d')[1])
    data['day'] = data['day'].astype(int)
    data['type'] = data.type.apply(lambda x: x.split('v')[1])
    data['type'] = data['type'].astype(int)

    user_set = np.unique(data['userid'])
    print "generate user"
    user_dic = {}
    for u in user_set:
        user_dic[u] = [[0]*49]*10

    print "generate data"
    for i in range(len(data.index)):
        if i%1000 == 0:
            print i
        t = data.type.iloc[i]
        w = data.week.iloc[i]
        d = data.day.iloc[i]
        user_dic[data.userid.iloc[i]][t-1][(w-1) * 7 + d-1] = data["count"].iloc[i]

    with open("user_dic.pkl", 'wb') as f:
        pickle.dump(user_dic, f, -1)


if __name__ == '__main__':
    start = time.time()
    flag = "submission" #submission, cross
    #preprocess(flag)
    train_model(flag)
    #linear_model(flag)
    #xgb_model(flag)
    #average_history()
    #average_predict()
    #fixed()
    #extract_user()
    #extract_data()

    print "Cost time is %s" %( (time.time() - start) / 1000 )
