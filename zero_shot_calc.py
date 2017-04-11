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

testid = 41
jdata_path = '/media/evl/Public/Mahyar/Data/CVPRdata/results41/logs/preds/test_predictions_5/*.json'
jclass_parse_path = '/media/evl/Public/Mahyar/Data/CVPRdata/batches12/test_class_parses_2933.json'
im_db = defaultdict(lambda: defaultdict(list))
ann_db = dict()
pred_db = dict()
weighted_classify = False

'''
Calculates top prediction class ids, among a set of class scores for a given image.
Input: a dict with class id as keys and list of (weight,score) as values
Output: top class id with highest score
'''
def find_prediction(pred):
    top = 1    
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
        wval.sort()
        vec = np.asarray(wval)
        res.append(np.mean(vec))
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
    preds, res = find_prediction(pred)
    pred_db[im] = preds
    gt = class_db[ann_db[im]]
    for p in preds:
        pc = class_db[p]
        cmat[gt, pc] += 1
        if p == ann_db[im]:
            acc += 1
            break

total = len(im_db.keys())
accuracy = acc / float(total)
print 'THIS IS HOW ACCURATE: '
print accuracy
print np.trace(cmat)*1.0 / total

'''
Plotting the confusion matrix.
'''
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
cax = ax.matshow(cmat)#, cmap='hot')
fig.colorbar(cax)

ax.set_xticklabels([0]+class_list, rotation=90)
ax.set_yticklabels([0]+class_list)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='both', which='minor', labelsize=6)
plt.grid(True, which='both')
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.savefig('/home/mahyar/confmat_'+str(testid)+'.pdf')
plt.savefig('/home/mahyar/confmat_'+str(testid)+'.png')
'''
### Example for sample display of a random normal
x = np.random.randn(1000)
y = np.random.randn(1000)+5
plt.hist2d(x, y, bins=40, cmap='hot')
plt.colorbar()
'''