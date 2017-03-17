# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:40:33 2016

@author: mahyarkoy
"""
import ex_generator as exgen
import scipy.io as sio
from collections import defaultdict
import numpy as np
import json
import glob
import os
import random

classes_path = '/media/evl/Public/Mahyar/Data/CVPRdata/CUB_200_2011/CUB_200_2011/classes.txt'
im_class_path = '/media/evl/Public/Mahyar/Data/CVPRdata/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
im_path = '/media/evl/Public/Mahyar/Data/CVPRdata/CUB_200_2011/CUB_200_2011/images.txt'
split_path = '/media/evl/Public/Mahyar/Data/CVPRdata/splits/train_test_split.mat'
parse_path = '/media/evl/Public/Mahyar/Data/CVPRdata/sps2_clean'
batch_path = '/media/evl/Public/Mahyar/Data/CVPRdata/batches13'
output_path = '/media/evl/Public/Mahyar/Data/CVPRdata/batches13'
log_path = '/media/evl/Public/Mahyar/Data/CVPRdata/batches13'

freq_dict = defaultdict(lambda: defaultdict(int))
train_idf_dict = dict()
test_idf_dict = dict()
except_list = ['(is (and this bird))', '(is (and bird particular))', '(is _thing)']
ref_db = list()
clusters = exgen.Clusters('cluster_info.pk')

def calc_freq():
    with open(classes_path) as cf: 
        for c in cf:
            cid = int(c.strip().split()[0])        
            cname = c.strip().split()[-1]
            ## make frequency list
            for fname in glob.iglob(parse_path+'/'+cname+'/*.sps2'):
                with open(fname) as fs:
                    for l in fs:
                        freq_dict[cid][l.strip()] += 1
                    

def calc_inverse_freq(cid_list):    
    idf_dict = dict()    
    parse_set = set()
    for c in cid_list:
        parse_set.update(freq_dict[c].keys())
    for ps in parse_set:
        count = 0
        for c in cid_list:
            k = freq_dict[c].keys()
            if ps in k:
                count += 1
        idf_dict[ps] = np.log(len(cid_list)/float(count)) if count > 0 else 0
    
    return idf_dict

def save_split(fname, data):
    temp = '%s %d'
    with open(fname, 'w+') as fo:
        for d in data:
            print>> fo, temp % (d['name'],d['cid'])
    return True
                    
def create_db_cub():
    im_db = list()
    imc_db = defaultdict(list)
    
    with open(im_path) as imf:    
        for im in imf:
            im_db.append(im.strip().split()[-1].strip('.jpg'))
            
    with open(im_class_path) as imcf:
        for imc in imcf:
            im_idx, label = map(int, imc.strip().split())
            imc_db[label].append(im_idx)
        
    with open(classes_path) as cf: 
        for c in cf:
            cid = int(c.strip().split()[0])        
            cname = c.strip().split()[-1]
            imlist = list(imc_db[cid])
            imnames = list(map(lambda x: im_db[x-1], imlist))
            ref_db.append({'id':cid, 'name': cname, 'im_ids':imlist, 'im_names': imnames})
    
    ### ref_db: {id: id of class, im_ids, im_names: list of all image file names, name: name of class}
    mat = sio.loadmat(split_path)
    train_cid_list = mat['train_cid'].tolist()[0]
    test_cid_list = mat['test_cid'].tolist()[0]
    train_db = list([ref_db[idx-1] for idx in train_cid_list])
    test_db = list([ref_db[idx-1] for idx in test_cid_list])
    
    print len(test_db)
    print len(train_db)
    
    train_set = list()
    test_set = list()
    for d in train_db:
        for i in range(len(d['im_ids'])):
            train_set.append({'id':d['im_ids'][i], 'name': d['im_names'][i], 'sname':d['im_names'][i],
                              'cid':d['id'], 'cname':d['name']})
            train_set.append({'id':d['im_ids'][i], 'name': d['im_names'][i]+'_fliph', 'sname':d['im_names'][i],
                              'cid':d['id'], 'cname':d['name']})

    for d in test_db:
        for i in range(len(d['im_ids'])):
            test_set.append({'id':d['im_ids'][i], 'name': d['im_names'][i], 'sname':d['im_names'][i],
                             'cid':d['id'], 'cname':d['name']})
            #test_set.append({'id':d['im_ids'][i], 'name': d['im_names'][i]+'_fliph', 'sname':d['im_names'][i],
            #                 'cid':d['id'], 'cname':d['name']})
    
    #save_split('/home/mahyar/cub_train_set_hard.txt', train_set)
    #save_split('/home/mahyar/cub_test_set_hard.txt', test_set)
    #print 'Splits are saved.'

    calc_freq()
    global train_idf_dict
    global test_idf_dict
    train_idf_dict = calc_inverse_freq(train_cid_list)
    test_idf_dict = calc_inverse_freq(test_cid_list)
    return train_set, test_set
    
###=======================NEG SAMPLING=========================###
NEG_SAMPLE = 10
POS_SAMPLE = 10
TEST_SAMPLE = 20

def normalize_features(pathname, data):
    # This is original calculation from training (***hardcode***)    
    mean = np.zeros((512,))
    mmt2 = np.zeros((512,))
    count = 0
    for di, d in enumerate(data):
        print 'AT IMAGE >>> '+str(di)
        with np.load(pathname+'/'+d['name'].split('/')[-1]+'.jpg.npz') as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]
            sq_image_data = np.square(image_data)
            mean += np.sum(image_data, axis=(1,2))
            mmt2 += np.sum(sq_image_data, axis=(1,2))
            count += image_data.shape[1] * image_data.shape[2]
    mean /= count
    mmt2 /= count
    var = mmt2 - np.square(mean)
    std = np.sqrt(var)

    # Save the mean std to file for future use (***hardcode***)
    np.savez('/media/evl/Public/Mahyar/Data/CVPRdata/normalizer_data_aug_hard.npz', mean=mean, std=std)    
      
def shuffle_list(parse_list):
    res = list(parse_list)
    np.random.shuffle(res)
    return res

def weighted_shuffle(parse_list, ci, idf_list=None):
    tf_thresh = 5
    score = [freq_dict[ci][x[0]] for x in parse_list]        
    if idf_list:
        flt = map(lambda x: x if x>=tf_thresh else 0, score)
        tf = np.array(flt)
        idf = np.array([idf_list[x[0]] for x in parse_list])
        sc = tf*idf
    else:
        sc = np.array(score)
    pr = sc + 0.001
    pr /= float(np.sum(pr))
    choice_ids = np.random.choice(np.arange(len(parse_list)), len(parse_list), replace=False, p=pr)
    choices = [parse_list[x] for x in choice_ids]
    return choices

def sort_list(parse_list, ci, idf_list=None, output_score=False):
    tf_thresh = 5
    score = [freq_dict[ci][x[0]] for x in parse_list]        
    if idf_list:
        flt = map(lambda x: x if x>=tf_thresh else 0, score)
        tf = np.array(flt)
        idf = np.array([idf_list[x[0]] for x in parse_list])
        sc = tf*idf
    else:
        sc = np.array(score)
    score_ids = np.argsort(sc).tolist()[::-1]
    score_vals = sc[score_ids]
    sort_res = [parse_list[x] for x in score_ids]
    if output_score:
        return sort_res, score_vals
    else:
        return sort_res

def read_parse(d, idf_list = None):
    dn = d['sname']
    parses = list()
    sents = list()
    parsef = parse_path + '/' + dn + '.sps2'
    sentf = parse_path + '/' + dn + '.sent'
    with open(parsef) as pf:
        for l in pf:
            parses.append(l.strip())
    with open(sentf) as sf:
        for l in sf:
            sents.append(l.strip())
    res = zip(parses, sents)
    if idf_list:
        ci = d['cid']
        sort_res = sort_list(res, ci, idf_list)
    else:
        sort_res = res
    return sort_res

def parse_cost(pi, ptrue):
    cost = 0
    if pi == ptrue:
        return 0
    spi = pi.replace('(','').replace(')','').split()
    sptrue = ptrue.replace('(','').replace(')','').split()
    if len(spi) > len(sptrue):
        large_parse = spi
        small_parse = sptrue
    else:
        large_parse = sptrue
        small_parse = spi
        
    for ws in small_parse:
        cost += 1        
        for wl in large_parse:
            if wl == ws:
                cost -= 1                
                break
    return cost
        
def find_negs(data, parses):
    threshold = 1
    negs = defaultdict(list)
    for di in range(len(data)):
        if len(negs[di]) >= NEG_SAMPLE:
            continue
        for dc in range(di+1, len(data)):
            if di == dc or data[di]['cid']==data[dc]['cid']:
                continue
            cost = np.zeros((len(parses[di]), len(parses[dc])))        
            for pi, piv in enumerate(parses[di]):
                for pc, pcv in enumerate(parses[dc]):
                    cost[pi, pc] = parse_cost(piv[0], pcv[0])
            
            mincost = np.min(cost, axis=1)
            minidx = np.where(np.logical_and(mincost <= threshold, mincost > 0))[0]
            for m in minidx:
                if parses[di][m][0] in except_list:
                    continue
                if len(negs[dc]) < NEG_SAMPLE:
                    for item in negs[dc]:
                        if item[0] == parses[di][m][0]:
                            break
                    else:
                        negs[dc].append(parses[di][m])
                else:
                    break
            
            mincost = np.min(cost, axis=0)
            minidx = np.where(np.logical_and(mincost <= threshold, mincost > 0))[0]
            #print(mincost[minidx])
            for m in minidx:
                if parses[dc][m][0] in except_list:
                    continue
                if len(negs[di]) < NEG_SAMPLE:
                    for item in negs[di]:
                        if item[0] == parses[dc][m][0]:
                            break
                    else:
                        negs[di].append(parses[dc][m])
                else:
                    break
    return negs

def find_negs_full(data, parses, idf_list = None):
    threshold = 1
    negs = defaultdict(list)
    for di in range(len(data)):
        datac_ids = shuffle_list(range(len(data)))
        for dc in datac_ids:
            if len(negs[di]) >= NEG_SAMPLE:
                break
            if di == dc or data[di]['cid']==data[dc]['cid']:
                continue
            if idf_list:
                ci = data[dc]['cid']
                choices = weighted_shuffle(parses[dc], ci)
            else:
                choices = parses[dc]

            for pc, pcv in enumerate(choices):
                min_cost = 10
                if len(negs[di]) >= NEG_SAMPLE:
                    break;
                if pcv[0] in except_list:
                    continue
                for pi, piv in enumerate(freq_dict[data[di]['cid']].keys()):
                    cost = parse_cost(piv, pcv[0])
                    if cost < min_cost:
                        min_cost = cost
                        #print di
                        #print (piv + '====' + pcv[0] + '-----' + str(cost))
                    if cost == 0:
                        break
                else:
                    if min_cost <= threshold:
                        for item in negs[di]:
                            if item[0] == pcv[0]:
                                break
                        else:
                            negs[di].append(pcv+(data[dc]['cid'],data[dc]['cname']))
    return negs

def find_pos(data, parses, get_all = False, idf_list = None):
    pos = defaultdict(list)
    for di in range(len(parses)):
        if idf_list:
            ci = data[di]['cid']
            choices = weighted_shuffle(parses[di], ci)
        else:
            choices = parses[di]
            
        for ps in choices:
            if ps[0] in except_list:
                continue           
            if get_all or len(pos[di]) < POS_SAMPLE:
                for item in pos[di]:
                    if item[0] == ps[0]:
                        break
                else:
                    pos[di].append(ps)
            else:
                break
    return pos
    
def make_batch_train(data, batch_size, fpath, idf_list):
    batch_id = 0    
    for batch_head in range(0, len(data), batch_size):
        print('AT BATCH >> '+ str(batch_id))
        batch_end = batch_head + batch_size
        batch_data = data[batch_head:batch_end]
        batch_set = list()
        batch_parses = list()
        batch_negs = list()
        for d in batch_data:
            batch_parses.append(read_parse(d))
        batch_pos = find_pos(batch_data, batch_parses, idf_list)
        batch_negs = find_negs_full(batch_data, batch_parses, idf_list)
        for idx, d in enumerate(batch_data):
            for ps in range(POS_SAMPLE):
                if ps >= len(batch_pos[idx]):
                    break
                batch_set.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                  'parse':batch_pos[idx][ps][0],
                                  'question':batch_pos[idx][ps][1], 'answer': 'yes',
                                  'cname':d['cname'], 'cid':d['cid'],
                                  'sent_cid':d['cid'], 'sent_cname':d['cname']})
            for ns in range(NEG_SAMPLE):
                if ns >= len(batch_negs[idx]):
                    break
                batch_set.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                  'parse':batch_negs[idx][ns][0],
                                  'question':batch_negs[idx][ns][1], 'answer': 'no',
                                  'cname':d['cname'], 'cid':d['cid'],
                                  'sent_cid':batch_negs[idx][ns][2], 'sent_cname':batch_negs[idx][ns][3]})
        
        with open(fpath+'/batch_'+str(batch_id)+'.json', 'w+') as fj:
            json.dump(batch_set, fj, indent=4)
        batch_id += 1

def select_neg_man(parse):
    return None
    word_freq_thresh = 100
    parse_words = parse.strip().replace(')', '').replace('(', '').split()
    watch_dog = 0
    while True:
        neg_word = clusters.get_negative(parse_words[-1])
        if not neg_word:
            return None
        if '-' in neg_word:
            neg_word = neg_word.split('-')[0]
        if train_word_count[neg_word] > word_freq_thresh:
            #print '----Accept'
            break
        watch_dog += 1
        if watch_dog > 10:
            return None

    parse_words[-1] = neg_word
    return '(is (and ' + ' '.join(parse_words[2:]) + '))'
    
def make_batch_train_man(data, batch_size, fpath, idf_list):
    batch_id = 0
    for batch_head in range(0, len(data), batch_size):
        print('AT BATCH >> '+ str(batch_id))
        batch_end = batch_head + batch_size
        batch_data = data[batch_head:batch_end]
        batch_output = list()
        selected_parses = select_parses(batch_data, idf_list)
        ### select top tf-idf scored parses for each image
        batch_pos = [parse_score_list[0] for parse_score_list in selected_parses]
        ### select neg for each parse by replacing last word, otherwise random pick
        batch_negs = [[] for i in range(len(batch_pos))]
        for loc, parse_sent_pairs in enumerate(batch_pos):
            for parse, sent in parse_sent_pairs:
                neg_parse = select_neg_man(parse)
                if not neg_parse:
                    while (True):
                        candid = np.random.random_integers(len(batch_pos)) - 1
                        if batch_data[candid]['cid'] != batch_data[loc]['cid']:
                            neg_parse = random.choice(batch_pos[candid][:10])
                            break
                else:
                    neg_parse = (neg_parse, sent)
                batch_negs[loc].append(neg_parse)
        ### construct and save batch_output
        for idx, d in enumerate(batch_data):
            for ps in range(POS_SAMPLE):
                if ps >= len(batch_pos[idx]):
                    break
                batch_output.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                  'parse':batch_pos[idx][ps][0],
                                  'question':batch_pos[idx][ps][1], 'answer': 'yes',
                                  'cname':d['cname'], 'cid':d['cid'],
                                  'sent_cid':d['cid'], 'sent_cname':d['cname']})
                ns = ps
                batch_output.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                  'parse':batch_negs[idx][ns][0],
                                  'question':batch_negs[idx][ns][1], 'answer': 'no',
                                  'cname':d['cname'], 'cid':d['cid'],
                                  'sent_cid':0, 'sent_cname':'unknown'})
        
        with open(fpath+'/batch_'+str(batch_id)+'.json', 'w+') as fj:
            json.dump(batch_output, fj, indent=4)
        batch_id += 1
        
def make_batch_train_contrastive(data, batch_size, fpath, idf_list):
    batch_id = 0
    ### each class holds a list of corresponding images data
    class_im_dict = defaultdict(list)
    for d in data:
        class_im_dict[d['cid']].append(d)
    ### shuffle each class list
    class_im_lists = class_im_dict.values()
    for im_list in class_im_lists:
        np.random.shuffle(im_list)     
    ### make candidate lists which contain unique data within
    max_len = len(max(class_im_lists, key=len))
    candid_lists = list()
    for peel in range(max_len):
        candid_lists.append([im_list[peel] if peel < len(im_list) else np.random.choice(im_list) for im_list in class_im_lists])
    ### shuffle each candidate list, then move over it and make batches
    for candid in candid_lists:
        np.random.shuffle(candid)
        for batch_head in range(0, len(candid), batch_size):
            print('AT BATCH >> '+ str(batch_id))
            batch_end = batch_head + batch_size
            batch_data = candid[batch_head:batch_end]
            batch_output = list()
            ### select top tf-idf scored parses for each image
            selected_parses = select_parses(batch_data, idf_list)
            def prob_over_range(length):
                return (1+np.arange(length)) * 2 / float(length*(length+1))
            batch_pos = [sp[0][np.random.choice(len(sp[0]), p=prob_over_range(len(sp[0]))[::-1])] for sp in selected_parses]
            for idx, d in enumerate(batch_data):
                batch_output.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                  'parse':batch_pos[idx][0],
                                  'question':batch_pos[idx][1], 'answer': 'yes',
                                  'cname':d['cname'], 'cid':d['cid'],
                                  'sent_cid':d['cid'], 'sent_cname':d['cname']})
            with open(fpath+'/batch_'+str(batch_id)+'.json', 'w+') as fj:
                json.dump(batch_output, fj, indent=4)
            batch_id += 1

def make_batch_test_contrastive(data, batch_size, fpath, idf_dict, sample_size=0):
    sample_size = len(data) if sample_size==0 else sample_size
    batch_id = 0
    class_parses = defaultdict(dict)
    class_parses_sorted = defaultdict(list)
    class_parses_scores = defaultdict(list)
    for d in data:
        parses = read_parse(d)
        for ps in parses:
            class_parses[d['cid']][ps[0]] = ps[1]
        
    for c in class_parses.keys():
        parses_sorted, scores = sort_list(class_parses[c].items(), c, idf_dict, True)
        for ps, score in zip(parses_sorted, scores):
            if ps[0] not in except_list:
                class_parses_sorted[c].append(ps)
                class_parses_scores[c].append(score)
            if len(class_parses_sorted[c]) >= TEST_SAMPLE:
                break
    
    c_batch = list()
    for c, c_data_list in class_parses_sorted.items():
        c_batch += zip(np.ones(len(c_data_list)).astype(int)*c, c_data_list)

    batch_id = 0
    for batch_head in range(0, sample_size, batch_size):
        batch_end = batch_head + batch_size
        batch_data = data[batch_head:batch_end]
        c_batch_size = len(batch_data)
        for c_batch_head in range(0, len(c_batch), c_batch_size):
            print('AT BATCH >> '+ str(batch_id) + ' == ' + str(c_batch_size))
            batch_set = list()
            c_batch_end = c_batch_head + c_batch_size
            c_batch_data = c_batch[c_batch_head:c_batch_end]
            if len(batch_data) > len(c_batch_data):
                for itr in range(len(c_batch_data), len(batch_data)):
                    c_batch_data.append((-1, ['(is (and _this _unknown))', 'unknown']))
            assert len(batch_data) == len(c_batch_data) ### This is stupid! should handle pairing better
            for d, ps in zip(batch_data, c_batch_data):
                ann = 'yes' if c==d['cid'] else 'no'
                batch_set.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                  'parse':ps[1][0],
                                  'question':ps[1][-1], 'answer': ann,
                                  'cname':d['cname'], 'cid':d['cid'],
                                  'sent_cid':ps[0], 'sent_cname':ref_db[ps[0]-1]['name']})
            with open(fpath+'/batch_'+str(batch_id)+'.json', 'w+') as fj:
                    json.dump(batch_set, fj, indent=4)
            batch_id += 1

    with open(log_path+'/test_class_parses_'+str(sample_size)+'.json', 'w+') as fj:
        json.dump(dict(zip(class_parses_sorted.keys(), \
                           zip(class_parses_sorted.values(), class_parses_scores.values()))), fj, indent=4)
    return class_parses_sorted
    
def make_batch_val(data, batch_size, fpath):
    batch_id = 0
    for batch_head in range(0, len(data), batch_size):
        print('AT BATCH >> '+ str(batch_id))
        batch_end = batch_head + batch_size
        batch_data = data[batch_head:batch_end]
        batch_set = list()
        batch_parses = list()
        for d in batch_data:
            batch_parses.append(read_parse(d))
        batch_pos = find_pos(batch_data, batch_parses, get_all=True)
        for idx, d in enumerate(batch_data):
            for ps in range(len(batch_pos[idx])):
                batch_set.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                  'parse':batch_pos[idx][ps][0],
                                  'question':batch_pos[idx][ps][-1], 'answer': 'yes',
                                  'cname':d['cname'], 'cid':d['cid'],
                                  'sent_cid':d['cid'], 'sent_cname':d['cname']})
        with open(fpath+'/batch_'+str(batch_id)+'.json', 'w+') as fj:
            json.dump(batch_set, fj, indent=4)
        batch_id += 1

def make_batch_test(data, batch_size, fpath, idf_dict, sample_size=0):
    sample_size = len(data) if sample_size==0 else sample_size
    batch_id = 0
    class_parses = defaultdict(dict)
    class_parses_sorted = defaultdict(list)
    class_parses_scores = defaultdict(list)
    for d in data:
        parses = read_parse(d)
        for ps in parses:
            class_parses[d['cid']][ps[0]] = ps[1]
        
    for c in class_parses.keys():
        parses_sorted, scores = sort_list(class_parses[c].items(), c, idf_dict, True)
        for ps, score in zip(parses_sorted, scores):
            if ps[0] not in except_list:
                class_parses_sorted[c].append(ps)
                class_parses_scores[c].append(score)
            if len(class_parses_sorted[c]) >= TEST_SAMPLE:
                break
        for batch_head in range(0, sample_size, batch_size):
            print('AT BATCH >> '+ str(batch_id))
            batch_end = batch_head + batch_size
            batch_data = data[batch_head:batch_end]
            batch_set = list()
            for idx, d in enumerate(batch_data):
                for psid, ps in enumerate(class_parses_sorted[c]):
                    ann = 'yes' if c==d['cid'] else 'no'
                    batch_set.append({'image':d['name'].split('/')[-1]+'.jpg.npz',
                                      'parse':ps[0],
                                      'question':ps[-1], 'answer': ann,
                                      'cname':d['cname'], 'cid':d['cid'],
                                      'sent_cid':c, 'sent_cname':ref_db[c-1]['name']})
            with open(fpath+'/batch_'+str(batch_id)+'.json', 'w+') as fj:
                json.dump(batch_set, fj, indent=4)
            batch_id += 1

    with open(log_path+'/test_class_parses_'+str(sample_size)+'.json', 'w+') as fj:
        json.dump(dict(zip(class_parses_sorted.keys(), \
                           zip(class_parses_sorted.values(), class_parses_scores.values()))), fj, indent=4)
    return class_parses_sorted

def select_parses(data_list, idf_dict):
    selected_parses = list()
    for data in data_list:
        unique_parses = list()
        unique_sents = list()
        for ps, sent in read_parse(data):
            if ps not in unique_parses:
                unique_parses.append(ps)
                unique_sents.append(sent)
        sorted_parse_pair_list, parse_scores = sort_list(zip(unique_parses, unique_sents), data['cid'], idf_dict, True)
        selected_parses.append((sorted_parse_pair_list, parse_scores))
    return selected_parses
    
if __name__ == '__main__':
    batch_size = 10
    num_itr = 100
    ### Read train and test set data
    train_set, test_set = create_db_cub()
    #selected_parses_test = select_parses(test_set, test_idf_dict)
    
    ### All adjectives
    #adj_set = set()
    #parse_set = set()
    parse_list = list()
    adj_list = list()
    sent_list = list()
    freq_list = list()
    for d in train_set:
        parse_tuples = read_parse(d)
        for ps, sent in parse_tuples:
            sps = ps.replace('(','').replace(')','').split()
            if ps not in except_list:
                #parse_set.update([ps])
                for loc in range(len(sps)):
                    if loc > 1:
                        term = sps[loc]
                        if term not in adj_list:
                            adj_list.append(term)
                            sent_list.append(sent)
                            parse_list.append(ps)
                            freq_list.append(1)
                        else:
                            freq_list[adj_list.index(term)] += 1
    '''
    with open('set_train_parses.sps','w+') as pf, open('set_train_adj.txt', 'w+') as af, open('set_train_sent.txt', 'w+') as sf:    
        for p in parse_list:
            print >>pf, p
        for a, f in zip(adj_list, freq_list):
            print >>af, ' '.join([a, str(f)])
        for s in sent_list:
            print >>sf, s
    '''  
    global train_word_count
    train_word_count = dict(zip(adj_list, freq_list))

    ### Make Validation batches
    fpath_val = batch_path + '/val'
    data_val = list(test_set)
    np.random.shuffle(data_val)
    os.system('mkdir '+ fpath_val)
    #make_batch_train_contrastive(data_val, batch_size*1, fpath_val, test_idf_dict)
    make_batch_train_man(data_val, batch_size, fpath_val, test_idf_dict)   
    
    ### Make Zero shot compare all batches for test data
    fpath_test = batch_path + '/test'
    data_test = list(test_set)
    #np.random.shuffle(data_val)
    os.system('mkdir '+ fpath_test)
    #class_parses = make_batch_test_contrastive(data_test, batch_size*10, fpath_test, test_idf_dict)
    class_parses = make_batch_test(data_test, batch_size*2, fpath_test, test_idf_dict)
    with open(output_path + '/test_class_parses.json', 'w+') as jf:
        json.dump(class_parses, jf, indent=4)
    
    ### Make Zero shot compare all batches for training data
    fpath_test_train = batch_path + '/test_train'
    data_test_train = list(train_set)
    np.random.shuffle(data_test_train)
    os.system('mkdir '+ fpath_test_train)
    #class_parses = make_batch_test_contrastive(data_test_train, batch_size*10, fpath_test_train, train_idf_dict, sample_size=1500)
    class_parses = make_batch_test(data_test_train, batch_size*2, fpath_test_train, train_idf_dict, sample_size=1500)
    with open(output_path + '/train_class_parses.json', 'w+') as jf:
        json.dump(class_parses, jf, indent=4)
    
    ### Make Training batches
    data_train = list(train_set)
    for itr in range(num_itr):
        print 'AT ITERATION ======== '+str(itr)
        np.random.shuffle(data_train)
        fpath_train = batch_path + '/itr_' + str(itr)
        os.system('mkdir ' + fpath_train)
        #make_batch_train_contrastive(data_train, batch_size*1, fpath_train, train_idf_dict)
        make_batch_train_man(data_train, batch_size, fpath_train, train_idf_dict)

    #normalize_features('/media/evl/Public/Mahyar/Data/CVPRdata/CUB_200_2011/CUB_200_2011/convs_aug', train_set)




















