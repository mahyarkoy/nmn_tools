# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:05:54 2016

@author: mahyarkoy
"""

import numpy as np
import json
from collections import defaultdict

jdata_path = '/media/evl/Public/Mahyar/Data/CVPRdata/results35/logs/test_predictions_5.json'
jclass_parse_path = '/media/evl/Public/Mahyar/Data/CVPRdata/batches12/test_class_parses_2933.json'
im_db = defaultdict(lambda: defaultdict(list))
ann_db = dict()
pred_db = dict()
weighted_classify = False

def find_prediction(pred):
    top = 1    
    #top_choice = 15
    res = list()
    pred_list = pred.items()
    for c, val in pred_list:
        #use top_choices to filter out noise, not useful currently
        wval = [v[0]*v[1] for v in val]
        wval.sort()
        vec = np.asarray(wval)
        res.append(np.mean(vec))
    cid = np.argsort(res)[::-1][0:top]
    #cid = np.random.choice(len(res),top)
    out = [pred_list[x][0] for x in cid.tolist()]
    return out, res

with open(jdata_path) as jf:
    jdata = json.load(jf)
with open(jclass_parse_path) as jf:
    class_parses = json.load(jf)

def get_class_parse_score(parse, cid):
    parse_pair_list = class_parses[str(cid)][0]
    score_list = class_parses[str(cid)][1]
    parse_list = [pp[0] for pp in parse_pair_list]
    idx = parse_list.index(parse) #throws error if not in list
    return  score_list[idx]
    
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

acc = 0
for im, pred in im_db.items():
    preds, res = find_prediction(pred)
    pred_db[im] = preds    
    for p in preds:
        if p == ann_db[im]:
            acc += 1
            break

total = len(im_db.keys())
accuracy = acc / float(total)
print 'THIS IS HOW ACCURATE: '
print accuracy