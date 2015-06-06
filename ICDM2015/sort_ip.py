import re
from operator import itemgetter
import csv


f = open('data/id_all_ip.csv')
rows = list()
r = re.compile('(\(\w+,\w+,\w+,\w+,\w+,\w+,\w+\))')
r2 = re.compile(',')
title = ['id', 'device_or_cookie', 'ips']

for t, i in enumerate(f):
    if t == 0:
        continue
    row = list()
    i = i.strip().split('{')
    row = i[0].split(',')[: -1]
    if row[1] == '1':
        continue
    ips = r.findall(i[1][: -1])
    s = ''
    for ip in ips:
        ip = r2.sub(' ', ip[1: -1]) + '|'
        s += ip
    s = s[:-1]
    row.append(s)
    rows.append(row)

f.close()
rows = sorted(rows, key=itemgetter(0))
output = open('data/device_ip.csv', 'wb')
open_file_object = csv.writer(output)
open_file_object.writerow(title)
open_file_object.writerows(rows)
output.close()
