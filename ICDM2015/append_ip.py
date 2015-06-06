import csv
import pandas as pd


train = pd.read_csv(r'data/dev_train_basic.csv', header=0).values
test = pd.read_csv(r'data/dev_test_basic.csv', header=0).values
cookie = pd.read_csv(r'data/cookie_all_basic.csv', header=0).values

device_ip_path = "data/device_ip.csv"
cookie_ip_path = "data/cookie_ip.csv"

dev_features = ['drawbridge_handle', 'device_id', 'device_type', 'device_os',
                'country', 'anonymous_c0', 'anonymous_c1', 'anonymous_c2',
                'anonymous_5', 'anonymous_6', 'anonymous_7', 'ips']
cookie_features = ['drawbridge_handle', 'cookie_id', 'computer_os_type',
                   'computer_browser_version', 'country', 'anonymous_c0',
                   'anonymous_c1', 'anonymous_c2', 'anonymous_5',
                   'anonymous_6', 'anonymous_7', 'ips']

train_output = open("data/train_ip_normalized.csv", "wb")
test_output = open("data/test_ip_normalized.csv", "wb")
cookie_output = open("data/cookie_ip_normalized.csv", "wb")

train_file_object = csv.writer(train_output)
test_file_object = csv.writer(test_output)
cookie_file_object = csv.writer(cookie_output)

train_file_object.writerow(dev_features)
test_file_object.writerow(dev_features)
cookie_file_object.writerow(cookie_features)

i = j = k = 0
train_rows = []
test_rows = []
cookie_rows = []

for t, dev in enumerate(open(device_ip_path)):
    if t == 0:
        continue
    dev = dev.strip().split(',')
    if dev[0] == train[i][1]:
        row = list(train[i])
        row.append(dev[2])
        train_rows.append(row)
        i += 1
    elif dev[0] == test[j][1]:
        row = list(test[j])
        row.append(dev[2])
        test_rows.append(row)
        j += 1

train_file_object.writerows(train_rows)
test_file_object.writerows(test_rows)
train_output.close()
test_output.close()

for t, c in enumerate(open(cookie_ip_path)):
    if t == 0:
        continue
    c = c.strip().split(',')
    if c[0] == cookie[k][1]:
        row = list(cookie[k])
        row.append(c[2])
        cookie_rows.append(row)
        k += 1
    if t % 10000 == 0 and t > 0:
        cookie_file_object.writerows(cookie_rows)
        cookie_rows = []
cookie_file_object.writerows(cookie_rows)
cookie_output.close()
print i
print len(train)
print j
print len(test)
print k
print len(cookie)
