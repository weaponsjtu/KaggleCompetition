import numpy as np
import pandas as pd

def feature_stat():
    train = pd.read_csv('data/train_ip_normalized.csv', header=0)
    test = pd.read_csv('data/test_ip_normalized.csv', header=0)
    cookie = pd.read_csv('data/cookie_ip_normalized.csv', header=0)

    device_cookie = pd.merge( train, cookie, on='drawbridge_handle')
    feature_ip = 0
    common_num = []
    for i in range(0, len(device_cookie.index)):
        device_ip = device_cookie['ips_x'].iat[i]
        cookie_ip = device_cookie['ips_y'].iat[i]
        tmp_commons = common_ips(device_ip, cookie_ip, 1)
        common_num.append(tmp_commons)
        if tmp_commons:
            feature_ip = feature_ip + 1
    df = pd.DataFrame(common_num, columns=['A'])
    print df['A'].value_counts()
    print feature_ip

# @submission
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

# @submission
def ips_common_sub():
    train = pd.read_csv('data/train_ip_normalized.csv', header=0)
    test = pd.read_csv('data/test_ip_normalized.csv', header=0)
    cookie = pd.read_csv('data/cookie_ip_normalized.csv', header=0)

    submission = open('submission_v4.csv', 'wb')
    submission.write('device_id,cookie_id\n')
    for i in range(0, len(test.index)):
        test_ip = test['ips'].iat[i]
        res = str(test['device_id'].iat[i]) + ','
        for j in range(0, len(cookie.index)):
            cookie_ip = cookie['ips'].iat[j]
            if common_ips(test_ip, cookie_ip) > 0:
                res = res + str(cookie['cookie_id'].iat[j]) + ' '
        res = res.strip() + '\n'
        submission.write(res)
    submission.close()

def common_ips(set1, set2, flag_p = 0):
    tmp1 = set1.split('|')
    for i in range(0, len(tmp1)):
        tmp1[i] = tmp1[i].split(' ')[0]
    tmp2 = set2.split('|')
    for i in range(0, len(tmp2)):
        tmp2[i] = tmp2[i].split(' ')[0]

    if flag_p != 0:
        print list(set(tmp1)&set(tmp2))

    return len( list(set(tmp1)&set(tmp2)) )


if __name__ == '__main__':
    #ip_common_sub()
    ips_common_sub()

    #feature_stat()
