import pandas as pd
import numpy as np
from sklearn import cross_validation, preprocessing
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import cPickle as pickle

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

def average_history():
    data = pd.read_csv('data/train.csv', sep='\t')

    features = ['userid', 'type', 'week', 'day']

    data['week'] = data.date.apply(lambda x: x.split('w')[1][:1])
    data['week'] = data['week'].astype(int)
    data['day'] = data.date.apply(lambda x: x.split('d')[1])
    data['day'] = data['day'].astype(int)
    data['type'] = data.type.apply(lambda x: x.split('v')[1])
    data['type'] = data['type'].astype(int)

    data = data.query('week >= 5 or (week == 5 & day >= 3)')
    user_set = np.unique(data.userid)

    #output = open("sub_avg.txt", "wb")
    #for i in range(len(user_set)):
    #    print i
    #    u = user_set[i]
    #    u_data = data.query('userid == @u')
    #    pred = []
    #    for d in range(1,8):
    #        for t in range(1,11):
    #            tmp = u_data.query('day == @d & type == @t')
    #            if len(tmp) > 0:
    #                average = round(sum(tmp['count']) * 1.0 / len(tmp))
    #            else:
    #                average = 0
    #            pred.append(int(average))
    #    if sum(pred) > 0:
    #        res = str(u) + '\t'
    #        for p in pred:
    #            res += str(p) + ','
    #        res = res[:-1]
    #        output.write(res + "\n")
    #output.close()

    # multi thread
    step = len(user_set) / multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    mp_list = []
    user_pred_list = {}
    for i in range(multiprocessing.cpu_count()):
        start = i * step
        end = (i+1)*step if (i+1)*step < len(user_set) else len(user_set)
        mp_data = data.copy()
        user_set_part = user_set[start:end].copy()
        user_pred_list[i] = manager.list()
        mp= ModelProcess(lock, user_set_part, user_pred_list[i], mp_data)
        mp_list.append(mp)

    for mp in mp_list:
        mp.start()

    for mp in mp_list:
        mp.join()

    output = open("sub_avg_mp.txt", "wb")
    for i in range(multiprocessing.cpu_count()):
        dic_part = user_pred_list[i]
        for s in dic_part:
            output.write(s + '\n')


def average_predict():
    data = pd.read_csv('data/train.csv', sep='\t')

    features = ['userid', 'type', 'week', 'day']

    data['week'] = data.date.apply(lambda x: x.split('w')[1][:1])
    data['week'] = data['week'].astype(int)
    data['day'] = data.date.apply(lambda x: x.split('d')[1])
    data['day'] = data['day'].astype(int)
    data['type'] = data.type.apply(lambda x: x.split('v')[1])
    data['type'] = data['type'].astype(int)

    train = data[ data['week'] < 7 ].copy()
    test = data[ data['week'] >= 7 ].copy()

    cate_keys = []
    for i in range(len(train.columns)):
        key = train.columns[i]
        if train.dtypes[key] == 'object':
            cate_keys.append(key)

    for key in cate_keys:
        #lbl = preprocessing.LabelEncoder()
        #lbl.fit( list(data[:, i]) )
        #train[:, i] = lbl.transform( list(train[:, i]) )
        #test[:, i] = lbl.transform( list(test[:, i]) )
        uniq = np.unique(train[key])
        train[key] = train[key].apply(lambda x: uniq.searchsorted(x))
        test[key] = test[key].apply(lambda x: uniq.searchsorted(x))

    print train.columns
    user_set = np.unique(train['userid'])

    print "generate label"
    label = np.zeros([len(user_set), 7*10])
    for i in range(len(test.index)):
        u = test['userid'].iloc[i]
        d = int(test['day'].iloc[i])
        t = int(test['type'].iloc[i])
        index = user_set.searchsorted(u)
        if index < len(user_set) and user_set[index] == u:
            label[index][(d-1)*10 + t - 1] = test['count'].iloc[i]

    print "generate prediction"
    train = train.query('week > 5 or (week == 5 & day >= 3)')
    print len(data)

    prediction = np.zeros([len(user_set), 7*10])
    for i in range(len(user_set)):
        u = user_set[i]
        print u
        u_data = train.query('userid == @u')
        pred = []
        for d in range(1,8):
            for t in range(1,11):
                tmp = u_data.query('type == @t & day == @d')
                if len(tmp) > 0:
                    average = round(sum(tmp['count']) * 1.0 / len(tmp))
                else:
                    average = 0
                pred.append(int(average))
        prediction[i] = pred

    with open('average_pred.pkl', 'wb') as f:
        pickle.dump([label, prediction], f, -1)

    print "score is %f" % (ml_metric(label, prediction))

import multiprocessing
class ModelProcess(multiprocessing.Process):
    def __init__(self, lock, user_set, user_pred, data):
        multiprocessing.Process.__init__(self)
        self.lock = lock
        self.user_pred = user_pred
        self.data = data
        self.user_set = user_set

    def run(self):
        for u in self.user_set:
            u_data = self.data.query('userid == @u')
            pred = []
            for d in range(1,8):
                for t in range(1,11):
                    tmp = u_data.query('day == @d & type == @t')
                    if len(tmp) > 0:
                        average = round(sum(tmp['count']) * 1.0 / len(tmp))
                    else:
                        average = 0
                    pred.append(int(average))

            if sum(pred) > 0:
                res = str(u) + '\t'
                for p in pred:
                    res += str(p) + ','
                res = res[:-1]
                self.user_pred.append(res)

if __name__ == '__main__':
    start = time.time()
    average_history()
    #average_predict()
    print "Cost time is %s" %( (time.time() - start) / 1000 )
