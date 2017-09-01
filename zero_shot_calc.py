# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:05:54 2016

@author: mahyarkoy
"""

import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cPickle as cpk
import glob

testname = 'gan_count_2'
#jdata_path = '/home/mahyar/cub_data/stack_gan_logs/logs/*.json'
#jdata_path = '/home/mahyar/CV_Res/nmn2/logs/preds/test_predictions_10/*.json'
#jdata_path = '/media/mahyar/My Passport/mahyar/gan_logs_1/logs/*.json'
jdata_path = '/home/mahyar/gan_logs_1/logs/*.json'
#jdata_path = '/media/evl/Public/Mahyar/Data/CVPRdata/results46/logs/preds/test_predictions_5/*.json'
#jdata_path = '/home/mahyar/CV_Res/koynmn/nmn2/logs/preds/test_predictions_5/*.json'
jclass_parse_path = '/media/evl/Public/Mahyar/Data/CVPRdata/batches12/test_class_parses_2933.json'
im_db = defaultdict(lambda: defaultdict(list))
ann_db = dict() 
pred_db = dict()
weighted_classify = False
res_db = dict()
ignore_list = [4, 9, 14, 29, 138, 38, 121, 166]

'''
Calculates top prediction class ids, among a set of class scores for a given image.
Input: a dict with class id as keys and list of (weight,score) as values
Output: top class id with highest score
'''
def find_prediction(pred):
    top = 10
    #top_choice = 15
    res = list()
    pred_list = pred.items()
    for c, val in pred_list:
        ### skip 0 class id: it is reserved for unknown class
        if c==0:
            res.append(-1.0)
            continue
        #use top_choices to filter out noise, not useful currently
        wval = [v[0]*v[1] for v in val]
        #wval.sort()
        vec = np.asarray(wval > 0)
        res.append(np.sum(vec))
    cid = np.argsort(res)[::-1][0:top]
    #cid = np.random.choice(len(res),top)
    out = [pred_list[x][0] for x in cid.tolist()]
    return out, res

'''
Reads parse and corresponding tf-idf scores from json file.
Input: string parse and its corresponding int class id
Output: score corresponding to given parse in the given class id.
Very inefficient.
'''
#with open(jclass_parse_path) as jf:
#    class_parses = json.load(jf)
class_parses = None
def get_class_parse_score(parse, cid):
    parse_pair_list = class_parses[str(cid)][0]
    score_list = class_parses[str(cid)][1]
    parse_list = [pp[0] for pp in parse_pair_list]
    idx = parse_list.index(parse) #throws error if not in list
    return  score_list[idx]

'''
For each image, stores scores for all classes seen in json files.
'''
class_set = set()
print "Filling im_db and class_set"
#with open(jdata_path) as jf:
#    jdata = json.load(jf)
#with open(jdata_path, 'rb') as jf:
#    jdata = cpk.load(jf)

for jpath in glob.glob(jdata_path):
    with open(jpath, 'r') as jf:
        jdata = json.load(jf)
        for jdict in jdata:
            imn = jdict['im_name']
            imc = jdict['im_cid']
            sc = jdict['sent_cid']
            yes_pr = jdict['prob']
            parse = jdict['parses']
            if weighted_classify == True:
                weight = get_class_parse_score(parse, sc)
            else:
                weight = 1.0
            im_db[imn][sc].append((yes_pr, weight))
            ann_db[imn] = imc
            class_set.add(imc)

print ">>> Class stats:"
print ">>>>>> num images: ", len(im_db.keys())
print ">>>>>> num text: ", np.asarray(im_db.values()).shape

'''Native accuracy'''
acc_db = defaultdict(list)
for imn, res in im_db.items():
    for sc, vals in res.items():
        key = 1 if sc == ann_db[imn] else -1
        acc_db[sc].append(np.mean([v[0]*key>0 for v in vals]))
acc_mat = np.array(acc_db.values())
acc_mean = np.mean(acc_mat,axis=1)
acc_std = np.std(acc_mat, axis=1)
sort_ids = np.argsort(acc_db.keys())

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.set_ylabel('Accuracy')
ax.set_title('Native Accuracy of ZSL')
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xticks(np.arange(50))
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.set_xticklabels(np.sort(acc_db.keys()))
bar1 = ax.bar(np.arange(len(acc_db)), acc_mean[sort_ids], yerr=acc_std[sort_ids])
plt.grid(True, which='both', linestyle='dotted')
plt.savefig('/home/mahyar/plots/native_acc_'+str(testname)+'.png')
vvv = ax.get_xticks()
print vvv

'''
Initializes class sorted list and confusion matrix
'''
print "Filling class_db and class_set"
acc = 0
class_count = len(class_set)
### confusion matrix construction
cmat = np.zeros((class_count,class_count))
class_db = dict()
class_list = list()
for ci, c in enumerate(np.sort(list(class_set))):
    class_db[c] = ci
    class_list.append(c)

'''
Calculates predictions and fills the confusion matrix.
'''
print "Filling cmat"            
for im, pred in im_db.items():
    if ann_db[im] in ignore_list:
        continue
    preds, res = find_prediction(pred)
    pred_db[im] = preds
    res_db[im] = res
    gt = class_db[ann_db[im]]
    for p in preds:
        if p in ignore_list:
            continue
        else:
            pc = class_db[p]
            cmat[gt, pc] += 1
            break

total = len(im_db.keys())
print 'THIS IS HOW ACCURATE: '
print np.trace(cmat)*1.0 / total

'''
Plotting the confusion matrix.
'''
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
cax = ax.matshow(cmat, cmap='jet')
fig.colorbar(cax)

ax.set_xticklabels([0]+class_list, rotation=90)
ax.set_yticklabels([0]+class_list)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='both', which='minor', labelsize=6)
plt.grid(True, which='both', linestyle='dotted')
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('True')
#plt.savefig('/home/mahyar/plots/confmat_'+str(testname)+'.pdf')
plt.savefig('/home/mahyar/plots/confmat_'+str(testname)+'.png')
'''
### Example for sample display of a random normal
x = np.random.randn(1000)
y = np.random.randn(1000)+5
plt.hist2d(x, y, bins=40, cmap='hot')
plt.colorbar()
'''
#138.Tree_Swallow/Tree_Swallow_0023_135345