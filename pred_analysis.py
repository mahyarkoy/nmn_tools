#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:55:43 2017
@author: mahyar

This script counts false positive, false negative,
and total occurances of each parse, in the given prediction file.
It also calculates the words occurances in each case, and generates output for d3.
The output shows total, false positive and false negative error of each word,
and also how much other words occumpanied that word in each case (contributed to error).
"""

import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import glob

batch_id = 40
test_path = '/media/evl/Public/Mahyar/Data/CVPRdata/results40/logs/test_predictions_5.json'
train_path = '/media/evl/Public/Mahyar/Data/CVPRdata/results40/logs/train_predictions_5.json'
save_d3_output = False
ignore_words = ['is', 'and']

### read validation prediction file
#with open(jdata_path, 'r') as jf:
#    jdata = json.load(jf)

def process_parse_db(db_path, word_list=None, filter_words=100):    
    '''
    Count false positive and negative and total occurances of each parse
    parse_db holds all parses as keys, and for each has word_list, false positive,
    and false negative count and total count.
    '''
    words_count_p = defaultdict(int)
    words_count_n = defaultdict(int)
    parse_db = defaultdict(lambda: defaultdict(int))
    for jfile in glob.glob(db_path):
        with open(jfile, 'r') as jf:
            jdata = json.load(jf)
            for d in jdata:
                parse = d['parses']
                im_cid = d['im_cid']
                sent_cid = d['sent_cid']
                ann = 1.0 if d['answer']=='yes' else 0.0
                gt = 1.0 if im_cid == sent_cid else 0.0
                parse_db[parse]['count'] += 1.0
                parse_db[parse]['parse'] = parse
                if gt:
                    parse_db[parse]['fn'] += 1.0 - ann
                    parse_db[parse]['p'] += 1.0
                else:
                    parse_db[parse]['fp'] += ann
                    parse_db[parse]['n'] += 1.0
                ### making words dictionary
                parse_words_list = list()
                parse_words = parse.strip().replace('(', '').replace(')','').split()
                for pw in parse_words:
                    if pw in ignore_words:
                        continue
                    words_count_p[pw] += gt
                    words_count_n[pw] += 1.0 - gt
                    parse_words_list.append(pw)
                parse_db[parse]['word_list'] = parse_words_list
                parse_db[parse]['word_list_total'] = parse_words_list
    
    '''
    Construct a list of filtered words, and a look up index dict. Also number of
    occurances of each word when positive ground truth and when false ground truth.
    We keep only the top 'top_words' words after sorting with total count of words.
    '''
    word_num_p = list()
    word_num_n = list()
    if not word_list:
        word_list = words_count_n.keys()
        top_words = len(word_list) if filter_words <= 0 else filter_words
        total_word_count = np.asarray(words_count_p.values()) + np.asarray(words_count_n.values())
        sorted_words_idx = np.argsort(total_word_count)[::-1]
        word_list = [word_list[idx] for idx in sorted_words_idx[:top_words]]
    
    for w in word_list:
        word_num_p.append(words_count_p[w])
        word_num_n.append(words_count_n[w])
    
    word_lookup = dict()
    for idx, w in enumerate(word_list):
        word_lookup[w] = idx
                   
    '''
    Filter out words from parse_db based on the total occurances of that words
    '''
    for p in parse_db.keys():
        if filter_words <= 0:
            break
        wl = list()
        for w in parse_db[p]['word_list']:
            if w in word_list:
                wl.append(w)
        parse_db[p]['word_list'] = wl
    
    '''
    Create two matrix of cooccurances for false positive and false negative.
    FP matrix shows error count on negative examples for each word on diag, and
    contribution of all words to that error off diag.
    FN matrix shows error count on positive examples for each word on diag, and 
    contribution of all words to that error off diag.
    '''
    #words_db = defaultdict(list)
    freq_mat_fp = np.zeros((len(word_list), len(word_list)))
    freq_mat_fn = np.zeros((len(word_list), len(word_list)))
    freq_mat_total = np.zeros((len(word_list), len(word_list)))
    freq_mat_p = np.zeros((len(word_list), len(word_list)))
    freq_mat_n = np.zeros((len(word_list), len(word_list)))
    for parse, freq in parse_db.items():
        parse_words = freq['word_list']
        for pw_i, pw in enumerate(parse_words):
            for pw2 in parse_words[pw_i:]:
                idx1 = word_lookup[pw]
                idx2 = word_lookup[pw2]
                
                freq_mat_fp[idx1, idx2] += freq['fp']
                freq_mat_fn[idx1, idx2] += freq['fn']
                freq_mat_total[idx1, idx2] += freq['count']
                freq_mat_p[idx1, idx2] += freq['p']
                freq_mat_n[idx1, idx2] += freq['n']
                
                if idx1 != idx2:
                    freq_mat_fp[idx2, idx1] += freq['fp']
                    freq_mat_fn[idx2, idx1] += freq['fn']
                    freq_mat_total[idx2, idx1] += freq['count']
                    freq_mat_p[idx2, idx1] += freq['p']
                    freq_mat_n[idx2, idx1] += freq['n']
    ### check symmetry. Note that these wont be symmetric after normalization.
    assert((freq_mat_fn.transpose() == freq_mat_fn).all())
        
    return parse_db, word_list, word_lookup, freq_mat_fp, freq_mat_fn, freq_mat_total, freq_mat_p, freq_mat_n

### find and print the parses with median, max and min errors  
'''                     
parse_list = parse_db.keys()
parse_errs = [(pdict['fp']+pdict['fn'])/float(pdict['count']) for pdict in parse_db.values()]
sorted_indices = np.argsort(parse_errs)
median_idx = sorted_indices[len(parse_errs)/2]
median_val = parse_errs[median_idx]
median_parse = parse_list[median_idx]
print 'MIN>>> ', parse_list[sorted_indices[0]], str(parse_errs[sorted_indices[0]])
print 'MEDIAN>>> ', median_parse, str(median_val)
print 'MAX>>> ', parse_list[sorted_indices[-1]], str(parse_errs[sorted_indices[-1]])
'''

'''
Normalize a given matrix:
1. divide diag elements by wnum vector (count of each word on each dim)
2. normalize off diag elements row wise

def normalize_mat(mat, wnum):
    mat = np.copy(mat)
    shape = mat.shape
    mat_diag = np.diag(np.diag(mat)) * 1.0 / wnum
    mat[range(shape[0]), range(shape[1])] = 0
    vec_sum = np.sum(mat, axis=1) + 0.001
    vec_sum = vec_sum.reshape((shape[0],1))
    mat_norm = mat / vec_sum
    out_mat = mat_norm + mat_diag
    #assert((out_mat.transpose() == out_mat).all())
    return out_mat
'''

'''
Normalize freq_mat s by the above normalization function.
FP_norm matrix shows error on negative examples for each word on diag, and
contribution of all words to that error off diag.
FN_norm matrix shows error on positive examples for each word on diag, and
contribution of all words to that error off diag.

wnum_n = np.array(word_num_n)
wnum_p = np.array(word_num_p)
'''

########## >>>>>MAIN<<<<< ##########
'''
Read train and test data
'''
bias_denum = 1e-6
parse_db, word_list, word_lookup, freq_mat_fp, freq_mat_fn, freq_mat_total, freq_mat_p, freq_mat_n = \
    process_parse_db(test_path, filter_words=80)
parse_db_tr, word_list_tr, word_lookup_tr, freq_mat_fp_tr, freq_mat_fn_tr, freq_mat_total_tr, freq_mat_p_tr, freq_mat_n_tr = \
    process_parse_db(train_path, word_list=word_list)
    
'''
Calculate error mat on total domain: positive and negative
'''

### N(X)
n_x = np.diag(freq_mat_total)
n_x = n_x.reshape(n_x.shape[0],1)
aug_n_x = np.sum(freq_mat_total, axis=1).reshape((freq_mat_total.shape[0],1)) - n_x

### N(X,Y,F)
freq_mat = freq_mat_fp + freq_mat_fn

### N(X,F)
n_x_f = np.diag(freq_mat)
n_x_f = n_x_f.reshape(n_x_f.shape[0], 1)
aug_n_x_f = np.sum(freq_mat, axis=1).reshape((freq_mat.shape[0],1)) - n_x_f

### P(F|X,Y) = N(X,Y,F) / N(X,Y)
freq_mat_norm = freq_mat / (freq_mat_total + bias_denum)
### P(Y|X,F) / P(Y|X) = ( N(X,Y,F) / N(X,F) ) / ( N(X,Y) / N(X) )
effect_mat = (freq_mat / (aug_n_x_f+bias_denum) + bias_denum) / (freq_mat_total / aug_n_x + bias_denum)

effect_mat[range(effect_mat.shape[0]), range(effect_mat.shape[0])] = 0
effect_mat += np.diag(np.diag(freq_mat_norm))

'''
Calculate error mat on positive domain
'''
### N(X,F,P)
n_x_f_p = np.diag(freq_mat_fn)
n_x_f_p = n_x_f_p.reshape(n_x_f_p.shape[0], 1)
aug_n_x_f_p = np.sum(freq_mat_fn, axis=1).reshape((freq_mat_fn.shape[0],1)) - n_x_f_p

### N(X,P)
n_x_p = np.diag(freq_mat_p)
n_x_p = n_x_p.reshape(n_x_p.shape[0], 1)
aug_n_x_p = np.sum(freq_mat_p, axis=1).reshape((freq_mat_p.shape[0],1)) - n_x_p

### P(F|X,Y,P) = N(X,Y,F,P) / N(X,Y,P)
freq_mat_norm_p = freq_mat_fn / (freq_mat_p + bias_denum)
### P(Y|X,F,P) / P(Y|X,P) = (N(X,Y,F,P) / N(X,F,P) ) / ( N(X,Y,P) / N(X,P) )
effect_mat_p = (freq_mat_fn / (aug_n_x_f_p+bias_denum) + bias_denum) / (freq_mat_p / (aug_n_x_p+bias_denum) + bias_denum)

effect_mat_p[range(effect_mat_p.shape[0]), range(effect_mat_p.shape[0])] = 0
effect_mat_p += np.diag(np.diag(freq_mat_norm_p))

'''
Calculate error mat on negative domain
'''
### N(X,F,N)
n_x_f_n = np.diag(freq_mat_fp)
n_x_f_n = n_x_f_n.reshape(n_x_f_n.shape[0], 1)
aug_n_x_f_n = np.sum(freq_mat_fp, axis=1).reshape((freq_mat_fn.shape[0],1)) - n_x_f_n
                  
### N(X,N)
n_x_n = np.diag(freq_mat_n)
n_x_n = n_x_n.reshape(n_x_n.shape[0], 1)
aug_n_x_n = np.sum(freq_mat_n, axis=1).reshape((freq_mat_n.shape[0],1)) - n_x_n

### P(F|X,Y,N) = N(X,Y,F,N) / N(X,Y,N)
freq_mat_norm_n = freq_mat_fp / (freq_mat_n + bias_denum)
### P(Y|X,F,N) / P(Y|X,N) = (N(X,Y,F,N) / N(X,F,N) ) / ( N(X,Y,N) / N(X,N) )
effect_mat_n = (freq_mat_fp / (aug_n_x_f_n+bias_denum) + bias_denum) / (freq_mat_n / (aug_n_x_n+bias_denum) + bias_denum)

effect_mat_n[range(effect_mat_n.shape[0]), range(effect_mat_n.shape[0])] = 0
effect_mat_n += np.diag(np.diag(freq_mat_norm_n))

'''
Calculate frequncy mat
'''
### P(Y|X) = N(X,Y) / N(X) off diag
dist_mat = freq_mat_total / (aug_n_x+bias_denum)
### P(x) = N(X) / N(all) on diag
dist_diag = n_x / (np.sum(n_x)+bias_denum)
dist_mat[range(dist_mat.shape[0]), range(dist_mat.shape[0])] = 0
dist_mat += np.diag(dist_diag[:,0])

'''
Calculate frequncy mat for training
'''
### N(X)
n_x_tr = np.diag(freq_mat_total_tr)
n_x_tr = n_x_tr.reshape(n_x_tr.shape[0],1)
aug_n_x_tr = np.sum(freq_mat_total_tr, axis=1).reshape((freq_mat_total_tr.shape[0],1)) - n_x_tr
### P(Y|X) = N(X,Y) / N(X) off diag
dist_mat_tr = freq_mat_total_tr / (aug_n_x_tr+bias_denum)
### P(x) = N(X) / N(all) on diag
dist_diag_tr = n_x_tr / (np.sum(n_x_tr)+bias_denum)
dist_mat_tr[range(dist_mat_tr.shape[0]), range(dist_mat_tr.shape[0])] = 0
#dist_mat_tr += np.diag(dist_diag_tr[:,0])

'''
Calculate entropy of each word, find correlation between entropy and error of
each word
'''
p_y_x_f = freq_mat / n_x
p_x_f = np.diag(p_y_x_f)

#p_y_x = freq_mat_total / aug_n_x
p_y_x = freq_mat_total_tr / (aug_n_x_tr+bias_denum)
p_y_x[range(p_y_x.shape[0]), range(p_y_x.shape[0])] = 0
h_y_x = -1.0 * np.log2(p_y_x+bias_denum).dot(p_y_x.transpose())
h_x = np.diag(h_y_x)# / n_x[:,0]

concat = np.asarray([p_x_f,h_x])
cov_mat = np.cov(concat)
corr = cov_mat[0,1] / (np.prod(np.diag(cov_mat))**0.5)
print 'Word Entropy and Error Correlation:'
print corr

word_freq_order = np.argsort(n_x_tr[:,0])[::-1]
plt.figure(0, figsize=(10,12))
plt.subplot(311)
plt.plot(p_x_f[word_freq_order], 'b')
plt.grid(True, which='both', linestyle='dotted')
plt.title('Word Error')

plt.subplot(312)
plt.plot(h_x[word_freq_order], 'r')
plt.grid(True, which='both', linestyle='dotted')
plt.title('Word Entropy')

plt.subplot(313)
plt.plot(n_x_tr[:,0][word_freq_order], 'm')
ax = plt.gca()
ax.set_yscale('log')
plt.grid(True, which='both', linestyle='dotted')
plt.title('Word Count')

plt.savefig('/home/mahyar/nmn_word_plot_'+str(batch_id)+'.pdf')

'''
Calculate correlation between variation and error of each parse
'''
### Sort parse_db values with frequency
parse_db_vals = parse_db_tr.values()
parse_db_counts = [v['count'] for v in parse_db_vals]
parse_db_vals = [parse_db_vals[i] for i in np.argsort(parse_db_counts)[::-1]]

### error of each parse
parse_err = list()
for v in parse_db_vals:
    parse_err.append( (v['fn']+v['fp'])*1.0 / v['count'])
parse_err = np.array(parse_err)

### variation of each parse: 1 - N(S) / N(S' ~ S)
parse_freq = np.zeros((parse_err.shape[0], parse_err.shape[0]))
for i, v in enumerate(parse_db_vals):
    for j in range(i, len(parse_db_vals)):
        if len( set(v['word_list_total']).intersection(set(parse_db_vals[j]['word_list_total'])) ) > 0:
            parse_freq[i,j] = parse_db_vals[j]['count']
            parse_freq[j,i] = v['count']

parse_freq_diag = np.diag(parse_freq)
parse_var = 1 - parse_freq_diag * 1.0 / np.sum(parse_freq, axis=1)# / parse_freq_diag
### correlation
parse_concat = np.asarray([parse_err, parse_var])
parse_cov_mat = np.cov(parse_concat)
parse_corr = parse_cov_mat[0,1] / (np.prod(np.diag(parse_cov_mat))**0.5)
print 'Parse Entropy and Error Correlation:'
print parse_corr

plt.figure(1,figsize=(10,12))
plt.subplot(311)
plt.plot(parse_err, 'b')
plt.title('Parse Error')
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 100))
plt.grid(True, which='both', linestyle='dotted')

plt.subplot(312)
plt.plot(parse_var, 'r')
plt.title('Parse Variations')
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 100))
plt.grid(True, which='both', linestyle='dotted')

plt.subplot(313)
plt.plot(parse_freq_diag, 'm')
plt.title('Parse Count')
ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 100))
ax.set_yscale('log')
plt.grid(True, which='both', linestyle='dotted')

plt.savefig('/home/mahyar/nmn_parse_plot_'+str(batch_id)+'.pdf')

'''
Calculate hellinger distance between test and train distribution
'''
hel_dist = np.sqrt(1 - np.sum(np.sqrt(dist_mat * dist_mat_tr), axis=1))
err_test = np.diag(freq_mat_norm)
dis_corr_concat = np.asarray([hel_dist, err_test])
dis_corr = np.corrcoef(dis_corr_concat)
print 'Correlation between the hel_dist and err_test:'
print dis_corr

hel_dist_total = np.sqrt(1 - np.sum(np.sqrt(dist_diag * dist_diag_tr)))
print 'Hellinger distance between marginal words distributions in test and training:'
print hel_dist_total

'''
### P(Y|X,F,N) = N(X,Y,F,N) / N(X,F,N) off diag
### P(F|X,N) = N(X,F,N) / N(X,N) on diag
freq_mat_fp_norm = normalize_mat(freq_mat_fp, wnum_n)

### P(Y|X,F,P) = N(X,Y,F,P) / N(X,F,P) off diag
### P(F|X,P) = N(X,F,P) / N(X,P) on diag
freq_mat_fn_norm = normalize_mat(freq_mat_fn, wnum_p)

### P(F|X,Y) = N(X,Y,F) / N(X,Y)
#freq_mat_norm = normalize_mat(freq_mat, wnum_n + wnum_p)
freq_mat_norm = freq_mat / (freq_mat_total + 0.00001)
'''

'''
Save as json to read by d3
'''

if save_d3_output:
    jdata_list = list()
    for idx, w in enumerate(word_list):
        freq_dict = dict()
        freq_dict['fp'] = effect_mat_n[idx,:].tolist()
        freq_dict['fn'] = effect_mat_p[idx,:].tolist()
        freq_dict['err'] = effect_mat[idx,:].tolist()
        freq_dict['dist'] = dist_mat[idx,:].tolist()
        jdata_list.append({'name': w, 'freq': freq_dict, 'id': idx, 'count': n_x[word_lookup[w]]})
        if idx == 9:
            f9 = freq_dict
            fx9 = freq_mat_fn[idx,:]
            w9 = w
    
    with open('/home/mahyar/nmn_words_stat_train_'+str(batch_id)+'.json', 'w+') as jf:
        json.dump({'nodes': jdata_list},jf)

