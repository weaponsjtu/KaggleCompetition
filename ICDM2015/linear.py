import numpy as np
import pandas as pd

def ip_common_sub():
    train = pd.read_csv('data/train_ip_normalized.csv', header=0)
    test = pd.read_csv('data/test_ip_normalized.csv', header=0)
    cookie = pd.read_csv('data/cookie_ip_normalized.csv', header=0)

    submission = open('submission_v3.csv', 'wb')
    submission.write('device_id,cookie_id\n')
    for i in range(0, len(test.index)):
        test_ip = test['ips'].iat[i]
        res = str(test['device_id'].iat[i]) + ','
        for j in range(0, len(cookie.index)):
            cookie_ip = cookie['ips'].iat[j]
            if common_ips(test_ip, cookie_ip) > 0:
                print j
                res = res + str(cookie['cookie_id'].iat[j]) +'\n'
                submission.write(res)
                break
    submission.close()

def common_ips(set1, set2):
    tmp1 = set1.split('|')
    for i in range(0, len(tmp1)):
        tmp1[i] = tmp1[i].split(' ')[0]
    tmp2 = set2.split('|')
    for i in range(0, len(tmp2)):
        tmp2[i] = tmp2[i].split(' ')[0]

    return len( list(set(tmp1)&set(tmp2)) )


if __name__ == '__main__':
    ip_common_sub()
