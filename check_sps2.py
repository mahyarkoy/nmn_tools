#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:25:26 2017

@author: mahyar
"""

import numpy as np
import os
from collections import defaultdict
import json
import matplotlib.pyplot as plt

dest_path = "/media/evl/Public/Mahyar/Data/CVPRdata/sps2_none"
log_path = "/media/evl/Public/Mahyar/Data/CVPRdata/sps2_none_log"

### class_db holds number of parse and sent file per class (folder) and corresponding fname_db
class_db = defaultdict(dict)
fname_db = defaultdict(dict)
for pname, dnames, fnames in os.walk(dest_path):
    if len(fnames) == 0:
        continue
    cname = pname.split('/')[-1]
    cid = int(cname.split('.')[0])
    class_db[cid]['file'] = fnames
    for fn in fnames:
        name, t = fn.split('.')
        with open(pname+'/'+fn, 'r') as fo:
            fdb = fname_db[name]
            fdb['cid'] = cid
            if t == 'sent':
                sentdb = defaultdict(int)
                for l in fo:
                    sentdb[l] += 1
                fdb['sent'] = dict(sentdb)
            if t == 'sps2':
                sps_set = set()
                for l in fo:
                    sps_set.add(l.strip())
                fdb['parse'] = set(sps_set)

file_vals = fname_db.values()
file_keys = fname_db.keys()
for i, cid in enumerate(np.sort(class_db.keys())):
    class_db[cid]['idx'] = i

parse_count = np.zeros(len(file_keys))
sent_count = np.zeros(len(file_keys))
class_parse_list = [set() for c in class_db.keys()]

err_db = dict()
for i, fv in enumerate(file_vals):
    flag = False
    parse_count[i] = len(fv['parse'])
    sent_count[i] = len(fv['sent'].values())
    class_parse_list[class_db[fv['cid']]['idx']].update(fv['parse'])
    for s in fv['sent'].values():
        ### Never happens!
        if s < 1:
            flag = True
    if flag or parse_count[i] < 10:
        err_db[file_keys[i]] = fv

class_parse_count = np.asarray([len(s) for s in class_parse_list])

os.system('mkdir '+log_path)
if len(err_db) > 0:
    print 'PRINTING ERROR...'
    with open(log_path+'/error.json', 'w+') as logf:
        json.dump(err_db, logf, indent=4)
        
plt.figure(0, figsize=(10,8))
plt.subplot(211)
plt.plot(sent_count, 'b')
plt.title('Sent Count')
plt.grid()

plt.subplot(212)
plt.plot(parse_count, 'r')
plt.title('Unique Parse Count')
plt.grid()

plt.savefig(log_path+'/file_plot.pdf')

plt.figure(1, figsize=(10,8))
plt.plot(class_parse_count, 'm')
plt.title('Class Unique Parse Count')
plt.grid()
plt.savefig(log_path+'/class_plot.pdf')
    

