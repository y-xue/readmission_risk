"""
Modified Clustering experiment for MGH lymphoma data.
Modified by yxue - 05-18-2017
First created by yluo - 01/26/2014
"""

#! /usr/bin/python;
import csv
import re
import numpy as np
import os
import sys
import subprocess
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.decomposition import PCA
from scipy.io import loadmat
from sklearn.svm import SVR, SVC
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, adjusted_mutual_info_score, roc_auc_score

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_util'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
import data_util as du
reload(du)

# cdn = '/data/mghcfl/mimic'
# cdn = '/Users/XueY/Desktop/summer project/SANMF/code'
# cdn = '/Users/XueY/Desktop/summer project/code/temporal-trends/last12h'
# cdn = '/Users/XueY/Documents/2016fall'

# cdn = '../data/raw'
# cdn = '../data/mean'
# cdn = '../data/mean(no results)'
# cdn = '../data/test'
# cdn = '../data/raw_shuffled'
# cdn = '../data/no_med_feature/raw_shuffled'
# cdn = '../data/no_med_feature/mean'


# pt_sg:
#   0 1 2 3 ...
#   --------
# 0|4 1 1 2
# 1|1 1 5 1
# 2|2 1 1 1
# 3|...
# pt1 has 5 sg2
def read_pt_sg_mat(fn):
    coo = np.genfromtxt(fn)
    # python is 0 based, but the file is 1 based
    pt_sg_mat = coo_matrix((coo[:,2], (coo[:,0]-1, coo[:,1]-1)))
    return np.array(pt_sg_mat.todense());

def read_sgs(fn):
    sgs = []
    f = open(fn)
    for ln in f:
        ln = ln.rstrip(" \n")
        sgs.append(ln)
    f.close()
    return sgs;

def sg_subiso(sgs):
    # detect subisomorphism among sgs, return adjacency matrix - 2 level hash
    hiso = {}
    # categorize sgs
    hsgc = {}; hsgstr = {}; hsgsize = {}
    for i in range(len(sgs)):
        sg = sgs[i]                 # sg = '5 (1)\tRSBI_n\t0 1 0 0 1 2\t[6]'
        sgrow = sg.split('\t'); 
        sgc = sgrow[1]              # sgc = 'RSBI_n'
        sgsize = int(re.sub(r'(\[|\])', "", sgrow[3]))
        if sgsize == 1:
            continue
        hsgsize[i] = sgsize # 6
        if not hsgc.has_key(sgc):
            hsgc[sgc] = []
        hsgc[sgc].append(i);        # hsgc: {'RSBI_n':[0,8,9,...]}
                                    # hsgsize[0] = 6, hsgsize[8] = 3, hsgsize[9] = 2
        hsgstr[i] = sgrow[2]        # hsgstr = '0 1 0 0 1 2'

    for sgc in hsgc:
        # sort subgraphs where value of nodes are sgc
        # by there hsgsize
        #
        # e.g. sgc = 'RSBI_n'
        # csgs = [9,8,0]
        csgs = sorted(hsgc[sgc], key=lambda sg: hsgsize[sg])

        # if sgc == "HCT_n":
        #     print(csgs)
        hsgc[sgc] = csgs
        for i in range(len(csgs)-1):
            for j in range(i+1,len(csgs)):
                sgi = csgs[i]; sgj = csgs[j]
                pati = '(?=%s)' % (hsgstr[sgi]); sgstrj = hsgstr[sgj]
                nmat = len(re.findall(pati, sgstrj))
                if nmat > 0:
                    # if sgc == 'HCT_n':
                    #     print('%s %s %s %s %s %s %s' % (sgc, i, j, sgi, sgj, pati, sgstrj))
                    
                    # add_2l_hash(h,k1,k2,v)
                    # h[k1][k2] += v
                    du.add_2l_hash(hiso, sgi, sgj, nmat)

    return (hiso, hsgstr, hsgc, hsgsize);

# ptid:
# 0|1778
# 1|1944
# 2|2040
# 3|...
# 
# ptid of pt0 is 1778
def read_pt_lab(fn):
    f = open(fn, 'r')
    pts = f.readlines()
    pts = map(lambda a: a.rstrip(" \n"), pts)
    if '' in pts:
        pts = pts.remove('')
    return pts;

def read_sparse_tensor(fn):
    spt = np.genfromtxt(fn)
    return spt;

def filter_sg(ptsg, hiso, sgs, spt=None):
    df_ptsg = pd.DataFrame(ptsg)
    for pi in range(df_ptsg.shape[0]):
        pt = df_ptsg.iloc[pi]
        hits = pt[pt>0].index.tolist()
        for hit in hits:
            if hiso.has_key(hit): # if subsumed by someone
                sgrow = sgs[hit].split('\t')
                sgnlab = len(set(sgrow[2].split()))
                sgsize = int(sgrow[3].strip("[]"))
                containers = set(hiso[hit].keys()).intersection(set(hits))
                for c in containers:
                    if not hiso.has_key(c) or not set(hiso[c].keys()).intersection(set(hits)): # if the maximal container 
                        ptsg[pi,hit] -= hiso[hit][c]
                        # if ptsg[pi,hit] < .5 and sgsize == 2: #  and sgnlab == 2
                        #     ptsg[pi,hit] = .5
                        # el
                        if ptsg[pi,hit] < 0:
                            # print('ptsg[%s, %s] < 0 with c: %s' % (pi, hit, c))
                            ptsg[pi,hit] = 0
    sgsum = np.sum(np.array(ptsg), axis=0)
    sgs = np.array(sgs)[sgsum>0]
    ptsg = ptsg[:,sgsum>0]

    # df_ptsg = pd.DataFrame(ptsg)
    # for pi in range(df_ptsg.shape[0]):
    #     pt = df_ptsg.iloc[pi]
    #     hits = pt[pt>0].index.tolist()
    #     for hit in hits:
    #         sgsize = int(sgs[hit].split('\t')[3].strip("[]"))
    #         ptsg[pi,hit] = sgsize
    if spt is not None:
        (sgsel,) = np.where(sgsum>0)
        hsgsel = {}
        for i in range(len(sgsel)):
            hsgsel[sgsel[i]+1] = i+1 # keep the 1-based index
        sptsel = spt[np.where(map(lambda x: hsgsel.has_key(int(x)), spt[:,1]))]
        sptsel[:,1] = map(lambda x: hsgsel[int(x)], sptsel[:,1])
        for i in range(sptsel.shape[0]):
            pi = sptsel[i,0]-1; sgi = sptsel[i,1]-1
            if ptsg[pi,sgi] == 0:
                sptsel[i,3] = 0
        sptsel = sptsel[np.where(sptsel[:,3]>0)]
        return (ptsg, sgs, sptsel);
    else:
        return (ptsg, sgs);

def filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 2, spt=None):
    """
    exclude patients whose subgraphs contain
    less than ntest_th different node's names (different features).
    """
    hsgc = {}
    df_ptsg = pd.DataFrame(ptsg)
    print 'len(sgs):', len(sgs)
    for i in range(len(sgs)):
        sg = sgs[i]
        sgrow = sg.split('\t'); sgc = sgrow[1]
        hsgc[i] = sgc
    
    sel = []
    for pi in range(len(pt)):
        ptrow = df_ptsg.iloc[pi]

        # for each subgraph in ptrow(a patient) 
        # if its cnt > 0, add the its node name to the set
        # e.g. set(['loc', 'Glucose_n', 'temp_n', 'mDBP_n', 'BUNtoCr_n'])
        tests = set(map(lambda k: hsgc[k], ptrow[ptrow>0].index.tolist()))

        tests = tests.difference(set(['loc'])) # e.g. set(['Glucose_n', 'temp_n', 'mDBP_n', 'BUNtoCr_n'])
        ntest = len(tests)
        if ntest >= ntest_th:
            sel.append(pi)
        else:
            print("exclude pt %s" % (pt[pi]))
    ptsgr = ptsg[sel,]
    ptr = np.array(pt)[sel]
    gtr = np.array(gt)[sel]
    ptwdr = ptwd[sel,]

    if spt!=None:
        hptsel = {}
        for i in range(len(sel)):
            hptsel[sel[i]+1] = i+1 # keep the 1-based index
        sptr = spt[np.where(map(lambda x: hptsel.has_key(int(x)), spt[:,0]))]
        sptr[:,0] = map(lambda x: hptsel[int(x)], sptr[:,0])
        return (ptsgr, ptwdr, ptr, gtr, sptr);
    else:
        return (ptsgr, ptwdr, ptr, gtr);

def read_pt_wd_mat(fn):
    coo = np.genfromtxt(fn)
    # python is 0 based, but the file is 1 based
    pt_wd_mat = coo_matrix((coo[:,2], (coo[:,0]-1, coo[:,1]-1)))
    # if not convert from matrix to array, then normalize will be inplace!
    return np.array(pt_wd_mat.todense());

# pt_gt:
#   0 1
# 0|1 0
# 1|1 0
# 2|0 1
# 3|...
# .....
# 
# outcome of pt2 is 1 (pt2 is predicted as readmitted)
def read_pt_gt(fn):
    coo = np.genfromtxt(fn)
    # hack
    pt_gt = coo_matrix((coo[:,1], (coo[:,0]-1, coo[:,2]-1)))
    return pt_gt.todense();

def split_tr_te_by_ptids(data, gt, pt, fntr, fnte):
    # fn = open(fnhpttid,'r')
    i = 0
    hpttid = {}
    for p in pt:
        hpttid[p] = i
        i += 1
    # for ln in fn:
    #     hpttid[int(ln)] = i
    #     i += 1
    # fn.close()

    # print hpttid

    train = pd.read_csv(fntr)
    test = pd.read_csv(fnte)

    # np.unique(train['sid'].tolist())

    train_index = np.unique(train['sid'].tolist())
    test_index = np.unique(test['sid'].tolist())

    filtered_tridlist = filter(lambda x: str(int(x)) in hpttid, train_index)
    filtered_teidlist = filter(lambda x: str(int(x)) in hpttid, test_index)

    # print filtered_teidlist

    tridlist = map(lambda x: hpttid[str(int(x))], filtered_tridlist)
    teidlist = map(lambda x: hpttid[str(int(x))], filtered_teidlist)

    # fperm = open('%s/myperm'%cdn, 'w')
    # for i in tridlist:
    #     fperm.write("%d\n" % i)
    # for i in teidlist:
    #     fperm.write("%d\n" % i)
    # fperm.close()

    # print teidlist

    data_tr = data[tridlist]
    data_te = data[teidlist]

    gt1c = np.argmax(gt,1)
    pgt = np.array(gt1c).ravel()
    gt_tr = pgt[tridlist]
    gt_te = pgt[teidlist]

    pt_tr = pt[tridlist]
    pt_te = pt[teidlist]

    return (data_tr, data_te, gt_tr, gt_te, pt_tr, pt_te)

def check_split_tr_te_by_ptids(fntr, fnte, data_tr, data_te, gt_tr, gt_te, pt_tr, pt_te):
    outcome_dic_tr = {}
    tr = pd.read_csv(fntr)
    gp = tr.groupby('sid')
    for sid, g in gp:
        outcome_dic_tr[sid] = g.iloc[0]['readmit']

    for i in range(len(gt_tr)):
        sid = pt_tr[i]
        oc = gt_tr[i]
        if outcome_dic_tr[int(sid)] != oc:
            exit('%d,%d'%(sid,i))

    outcome_dic_te = {}
    te = pd.read_csv(fnte)
    gp = te.groupby('sid')
    for sid, g in gp:
        outcome_dic_te[sid] = g.iloc[0]['readmit']

    for i in range(len(gt_te)):
        sid = pt_te[i]
        oc = gt_te[i]
        if outcome_dic_te[int(sid)] != oc:
            exit('%d,%d'%(sid,i))

import pickle
def functionality_check():
    folder = '../observer/seed2222_raw_mice_mp0.5mc0.6_cz/isg0/pt_sg_w'
    for stt in ['z','cz']:
        for freq_t in ['001','002','003','004','005','006','008','009', '01','011']:
            for i in range(5):
                fntr = '../data/seed2222/raw/mice/mp0.5_mc0.6/dataset/train_fold%s_%s.csv'%(i,stt)
                fnte = '../data/seed2222/raw/mice/mp0.5_mc0.6/dataset/test_fold%s_%s.csv'%(i,stt)
                ptsg = pickle.load(open('%s/ptsg_%s_fold%d'%(folder,freq_t,i),'rb'))
                pt = pickle.load(open('%s/pt_%s_fold%d'%(folder,freq_t,i),'rb'))
                ptwd = pickle.load(open('%s/ptwd_%s_fold%d'%(folder,freq_t,i),'rb'))
                sgs = pickle.load(open('%s/sgs_%s_fold%d'%(folder,freq_t,i),'rb'))
                gt = pickle.load(open('%s/gt_%s_fold%d'%(folder,freq_t,i),'rb'))
                (ptsg, ptwd, pt, gt) = filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 2)
                (data_tr, data_te, gt_tr, gt_te, pt_tr, pt_te) = split_tr_te_by_ptids(
                    ptsg, gt, pt, fntr, fnte)
                check_split_tr_te_by_ptids(
                    fntr, fnte, data_tr, data_te, gt_tr, gt_te, pt_tr, pt_te)

# functionality_check()

def sanity_check(fnsg='/data/mghcfl/mimic/data/pt_sg_w/mimic.sgstr', fnwd='/data/mghcfl/mimic/data/pt_sg_w/mimic.ntid'):
    fsg = open(fnsg)
    hsg = {}
    lc = -1
    while 1:
        ln = fsg.readline()
        if not ln:
            break
        lc += 1
        if not re.search(r'\[2', ln):
            m = re.search(r'(\S+_n_\S+)\]', ln)
            if m:
                wd = m.group(1)
                hsg[wd] = lc
    fsg.close()
    fwd = open(fnwd)
    lc = -1
    lwd = []
    while 1:
        ln = fwd.readline()
        if not ln:
            break
        lc += 1
        ln = ln.rstrip(" \n")
        if len(ln) > 0:
            lwd.append(ln)
    
    return (hsg, lwd);
