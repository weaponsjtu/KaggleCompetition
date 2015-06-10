import numpy as np
import pandas as pd


def feature():
    train = pd.read_csv('data/train_ip_property.csv', header=0)
    cookie = pd.read_csv('data/cookie_ip_property.csv', header=0)
    device_cookie = pd.merge( train, cookie, on='drawbridge_handle')

    unique_device_type = np.unique(device_cookie['device_type'])
    unique_device_os = np.unique(device_cookie['device_os'])
    unique_country_x = np.unique(device_cookie['country_x'])
    unique_a_c0_x = np.unique(device_cookie['anonymous_c0_x'])
    unique_a_c1_x = np.unique(device_cookie['anonymous_c1_x'])
    unique_a_c2_x = np.unique(device_cookie['anonymous_c2_x'])
    unique_a_5_x = np.unique(device_cookie['anonymous_5_x'])
    unique_a_6_x = np.unique(device_cookie['anonymous_6_x'])
    unique_a_7_x = np.unique(device_cookie['anonymous_7_x'])

    unique_computer_os_type = np.unique(device_cookie['computer_os_type'])
    unique_computer_browser_version = np.unique(device_cookie['computer_browser_version'])
    unique_country_y = np.unique(device_cookie['country_y'])
    unique_a_c0_y = np.unique(device_cookie['anonymous_c0_y'])
    unique_a_c1_y = np.unique(device_cookie['anonymous_c1_y'])
    unique_a_c2_y = np.unique(device_cookie['anonymous_c2_y'])
    unique_a_5_y = np.unique(device_cookie['anonymous_5_y'])
    unique_a_6_y = np.unique(device_cookie['anonymous_6_y'])
    unique_a_7_y = np.unique(device_cookie['anonymous_7_y'])

    n = len(device_cookie)
    for i in range(0, n):
        device_cookie['device_type'].iat[i] = value2vec( device_cookie['device_type'].iat[i], unique_device_type )
        device_cookie['device_os'].iat[i] = value2vec( device_cookie['device_os'].iat[i], unique_device_os)
        device_cookie['country_x'].iat[i] = value2vec( device_cookie['country_x'].iat[i], unique_country_x )
        device_cookie['anonymous_c0_x'].iat[i] = value2vec( device_cookie['anonymous_c0_x'].iat[i], unique_a_c0_x )
        device_cookie['anonymous_c1_x'].iat[i] = value2vec( device_cookie['anonymous_c1_x'].iat[i], unique_a_c1_x )
        device_cookie['anonymous_c2_x'].iat[i] = value2vec( device_cookie['anonymous_c2_x'].iat[i], unique_a_c2_x )
        device_cookie['anonymous_5_x'].iat[i] = float(device_cookie['anonymous_5_x'].iat[i]) * 1.0 / unique_a_5_x.max()
        device_cookie['anonymous_6_x'].iat[i] = float(device_cookie['anonymous_6_x'].iat[i]) * 1.0 / unique_a_6_x.max()
        device_cookie['anonymous_7_x'].iat[i] = float(device_cookie['anonymous_7_x'].iat[i]) * 1.0 / unique_a_7_x.max()

        device_cookie['computer_os_type'].iat[i] = value2vec( device_cookie['computer_os_type'].iat[i], unique_computer_os_type )
        device_cookie['computer_browser_version'].iat[i] = value2vec( device_cookie['computer_browser_version'].iat[i], unique_computer_browser_version )
        device_cookie['country_y'].iat[i] = value2vec( device_cookie['country_y'].iat[i], unique_country_y )
        device_cookie['anonymous_c0_y'].iat[i] = value2vec( device_cookie['anonymous_c0_y'].iat[i], unique_a_c0_y )
        device_cookie['anonymous_c1_y'].iat[i] = value2vec( device_cookie['anonymous_c1_y'].iat[i], unique_a_c1_y )
        device_cookie['anonymous_c2_y'].iat[i] = value2vec( device_cookie['anonymous_c2_y'].iat[i], unique_a_c2_y )
        device_cookie['anonymous_5_y'].iat[i] = float(device_cookie['anonymous_5_y'].iat[i]) * 1.0 / unique_a_5_y.max()
        device_cookie['anonymous_6_y'].iat[i] = float(device_cookie['anonymous_6_y'].iat[i]) * 1.0 / unique_a_6_y.max()
        device_cookie['anonymous_7_y'].iat[i] = float(device_cookie['anonymous_7_y'].iat[i]) * 1.0 / unique_a_7_y.max()

    device_cookie.to_csv('device_cookie_feature.csv')

def value2vec(value, array):
    res = [0] * len(array)
    res[ array.searchsorted(value) ] = 1
    string = ''
    for i in res:
        string = string + str(i) + ' '
    return string.strip()

def country_stat():
    train = pd.read_csv('data/train_ip_normalized.csv', header=0)
    cookie = pd.read_csv('data/cookie_ip_normalized.csv', header=0)
    device_cookie = pd.merge( train, cookie, on='drawbridge_handle')

    same_country = 0
    for i in range(0, len(device_cookie.index)):
        if device_cookie['country_x'].iat[i] == device_cookie['country_y'].iat[i]:
            same_country = same_country + 1
    print same_country


def cellular_stat():
    train = pd.read_csv('data/train_ip_normalized.csv', header=0)
    test = pd.read_csv('data/test_ip_normalized.csv', header=0)
    cookie = pd.read_csv('data/cookie_ip_normalized.csv', header=0)
    ipagg = pd.read_csv('data/ipagg_all.csv', header=0)

    cellular_num = []
    for i in range(0, len(train.index)):
        ips = train['ips'].iat[i]
        ips = ips.split('|')
        cellular = 0
        for ip in ips:
            if ipagg[ ipagg['ip_address'] == ip.split(' ')[0] ].iat[0, 1] == 1:
                cellular = cellular + 1
        cellular_num.append(cellular)

    df = pd.DataFrame(cellular_num, columns=['A'])
    print df['A'].value_counts()


def feature_stat():
    train = pd.read_csv('data/train_ip_normalized.csv', header=0)
    test = pd.read_csv('data/test_ip_normalized.csv', header=0)
    cookie = pd.read_csv('data/cookie_ip_normalized.csv', header=0)
    ipagg = pd.read_csv('data/ipagg_all.csv', header=0)
    device_cookie = pd.merge( train, cookie, on='drawbridge_handle')

    cellular_ip = 0
    feature_ip = 0
    common_num = []
    for i in range(0, len(device_cookie.index)):
        device_ip = device_cookie['ips_x'].iat[i]
        cookie_ip = device_cookie['ips_y'].iat[i]
        tmp_commons = common_ips(device_ip, cookie_ip)
        for j in range(0, len(tmp_commons)):
            df_ip = ipagg[ ipagg['ip_address'] == tmp_commons[j] ]
            if df_ip.iat[0,1] == 1:
                cellular_ip = cellular_ip + 1
                break

        common_num.append(len(tmp_commons))
        if len(tmp_commons):
            feature_ip = feature_ip + 1
    df = pd.DataFrame(common_num, columns=['A'])
    print df['A'].value_counts()
    print feature_ip
    print cellular_ip

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
            if len(common_ips(test_ip, cookie_ip)) > 0:
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
            if len(common_ips(test_ip, cookie_ip)) > 0:
                res = res + str(cookie['cookie_id'].iat[j]) + ' '
        res = res.strip() + '\n'
        submission.write(res)
    submission.close()

def common_ips(set1, set2):
    tmp1 = set1.split('|')
    for i in range(0, len(tmp1)):
        tmp1[i] = tmp1[i].split(' ')[0]
    tmp2 = set2.split('|')
    for i in range(0, len(tmp2)):
        tmp2[i] = tmp2[i].split(' ')[0]

    return list(set(tmp1)&set(tmp2))


if __name__ == '__main__':
    #ip_common_sub()
    #ips_common_sub()

    #feature_stat()
    #cellular_stat()
    #country_stat()

    feature()
