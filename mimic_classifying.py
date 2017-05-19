import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import RFE
from datetime import datetime
import time

import sklearn_util as sklu
reload(sklu)

def directClassify(mtr, mte, gt_tr, gt_te, clf=None, norm=None, fnroc=None, mean_impute=False):
    if clf == None:
        clf = LogisticRegression(penalty='l2', class_weight='balanced')
    if mean_impute:
        mtr = sklu.mean_imputing(mtr)
        mte = sklu.mean_imputing(mte)

    res = sklu.classify(mtr, mte, gt_tr, gt_te, clf, prob=True, norm=norm)
    if fnroc != None:
        ln = len(res['fpr_te'])
        roc = np.hstack((res['fpr_te'].reshape(ln,1), res['tpr_te'].reshape(ln,1)))
        rocdf = pd.DataFrame(roc, columns = ['fpr_te', 'tpr_te'])
        rocdf.to_csv(fnroc,index=False)
    return res;

def directClassify_cv(data, gt, pt, train_index, test_index, clf=None, norm=None, fnroc=None, mean_impute=False):
    mtr = data[train_index]; gt_tr = gt[train_index]
    mte = data[test_index]; gt_te = gt[test_index]
    if clf == None:
        clf = LogisticRegression(penalty='l1', class_weight='balanced')
    if mean_impute:
        mtr = sklu.mean_imputing(mtr)
        mte = sklu.mean_imputing(mte)

    res = sklu.classify(mtr, mte, gt_tr, gt_te, clf, prob=True, norm=norm)
    if fnroc != None:
        ln = len(res['fpr_te'])
        roc = np.hstack((res['fpr_te'].reshape(ln,1), res['tpr_te'].reshape(ln,1)))
        rocdf = pd.DataFrame(roc, columns = ['fpr_te', 'tpr_te'])
        rocdf.to_csv(fnroc,index=False)
    return res;

def nmfClassify(data_tr, data_te, gt_tr, gt_te, pt_tr, pt_te, nc=10, norm=None, fnfmt=None, clf=None, header="", fnroc=None, fncoef=None, fngrp=None, sgs=None):
    fn = fnfmt
    print fn
    if os.path.isfile(fn):
        print 'load %s' % fn
        with open(fn, 'rb') as f:
            [m, mtr, mte] = pickle.load(f)
    else:
        # t1 = datetime.now()
        start = time.time()
        m = NMF(nc, init='nndsvd', sparseness='components', eta=2, random_state=2222)
        m.fit(data_tr)
        mtr = m.transform(data_tr)
        mte = m.transform(data_te)

        td = time.time() - start
        # t2 = datetime.now()
        # td = t2 - t1
        # print 'NMF takes %f secs, reconstruction error %f' % (td.total_seconds(), m.reconstruction_err_)
        print 'NMF takes %f secs, reconstruction error %f' % (td, m.reconstruction_err_)

        with open(fn, 'wb') as f:
            pickle.dump([m, mtr, mte], f, -1)

    print '# positive in test: %d' % sum(gt_te)
    if clf == None:
        clf = LogisticRegression(penalty='l2', class_weight='balanced')
    res = sklu.classify(mtr, mte, gt_tr, gt_te, clf, prob=True, norm=norm, header='%snmf%d'%(header,nc))
    if fnroc != None:
        n = len(res['fpr_te'])
        roc = np.hstack((res['fpr_te'].reshape(ln,1), res['tpr_te'].reshape(ln,1)))
        rocdf = pd.DataFrame(roc, columns = ['fpr_te', 'tpr_te'])
        rocdf.to_csv(fnroc,index=False)

    coef = pd.DataFrame(np.transpose(clf.coef_))
    if fncoef != None:
        coef.to_csv(fncoef,index=False)
    if fngrp != None:
        coef_desc = coef.sort(columns=0,ascending=False)
        coef_ascd = coef.sort(columns=0)
        components = m.components_
        # sgs = md.read_sgs()
        grpt = 10; sgt = 20
        fgrp = open(fngrp, 'w')
        fgrp.write("Class 1\n")
        for grpi in range(grpt):
            grpid = coef_desc.index[grpi]
            fgrp.write("\n%d group # %d\n" % (grpi+1, grpid))
            grp = pd.DataFrame(components[grpid,])
            grp_sort = grp.sort(columns=0, ascending=False) / grp.sum()
            sgcnt = 0; sgi = 0
            while sgcnt < sgt:
                sgid = grp_sort.index[sgi]
                sgi += 1
                if sgid < len(sgs):
                    fgrp.write("%.4f\t%s\n" % (grp_sort.iloc[sgi,0], sgs[sgid]))
                    sgcnt += 1
        fgrp.write("\nClass 0\n")
        for grpi in range(grpt):
            grpid = coef_ascd.index[grpi]
            fgrp.write("\n%d group # %d\n" % (grpi+1, grpid))
            grp = pd.DataFrame(components[grpid,])
            grp_sort = grp.sort(columns=0, ascending=False) / grp.sum()
            sgcnt = 0; sgi = 0
            while sgcnt < sgt:
                sgid = grp_sort.index[sgi]
                sgi += 1
                if sgid < len(sgs):
                    fgrp.write("%.4f\t%s\n" % (grp_sort.iloc[sgi,0], sgs[sgid]))
                    sgcnt += 1
        fgrp.close()
    return (m, clf, res);

def nmfClassify_addMoreFeature(data_tr, data_te, gt_tr, gt_te, pt_tr, pt_te, nc=10, norm=None, fnpik=None, clf=None, header="", fnroc=None, fncoef=None, fngrp=None, sgs=None, fnaddtr=None, fnaddte=None, selected_features=None, foldi=0):
    print fnpik
    if os.path.isfile(fnpik):
        print 'load %s' % fnpik
        with open(fnpik, 'rb') as f:
            [m, mtr, mte] = pickle.load(f)
        
        train = pd.read_csv(fnaddtr)
        test = pd.read_csv(fnaddte)
        train_index = np.unique(train['sid'].tolist())
        test_index = np.unique(test['sid'].tolist())
        # train = train.drop(['sid','timeindex'],axis=1)
        # test = test.drop(['sid','timeindex'],axis=1)
        if selected_features != None:
            train = train[selected_features]
            test = test[selected_features]
        tr = np.array(train)
        te = np.array(test)
        if len(train_index) != len(tr) or len(test_index) != len(te):
            exit('interesting.')

        pt_tr_add = {}
        for i in range(len(train_index)):
            pt_tr_add[train_index[i]] = i
        pt_te_add = {}
        for i in range(len(test_index)):
            pt_te_add[test_index[i]] = i

        # print train_index
        # print test_index
        # print pt_tr
        # print pt_te

        tridlist = map(lambda x:pt_tr_add[int(x)],pt_tr)
        teidlist = map(lambda x:pt_te_add[int(x)],pt_te)

        # tridlist1 = []
        # teidlist1 = []
        # j = 0
        # for i in range(len(train_index)):
        #     if j >= len(pt_tr):
        #         print "here"
        #         break
        #     if train_index[i] == int(pt_tr[j]):
        #         tridlist1.append(i)
        #         j += 1

        # j = 0
        # for i in range(len(test_index)):
        #     if j >= len(pt_te):
        #         print "or here"
        #         break
        #     if test_index[i] == int(pt_te[j]):
        #         teidlist1.append(i)
        #         j += 1

        # if tridlist != tridlist1 or teidlist != teidlist1:
        #     exit('look')

        tr = tr[tridlist]
        te = te[teidlist]

        res_baseline = sklu.classify(tr, te, gt_tr, gt_te, LogisticRegression(penalty='l1', class_weight='balanced'), prob=True, norm=norm, header='%sbaseline'%(header))

        print mtr.shape
        print mte.shape
        mtr = np.hstack([mtr,tr])
        mte = np.hstack([mte,te])
        print mtr.shape
        print mte.shape

        # pd.DataFrame(mtr).to_csv('../observer/error_analysis/temporal+baseline_train_fold%d.csv'%foldi)
        # pd.DataFrame(mte).to_csv('../observer/error_analysis/temporal+baseline_test_fold%d.csv'%foldi)

    print '# positive in test: %d' % sum(gt_te)
    if clf == None:
        clf = LogisticRegression(penalty='l2', class_weight='balanced')
    res = sklu.classify(mtr, mte, gt_tr, gt_te, clf, prob=True, norm=norm, header='%snmf%d'%(header,nc))
    # print clf.coef_
    return (m, clf, res, res_baseline)

def dirClassify_addMoreFeature(data_tr, data_te, gt_tr, gt_te, pt_tr, pt_te, norm=None, clf=None, header="", fnroc=None, fncoef=None, fngrp=None, sgs=None, fnaddtr=None, fnaddte=None, selected_features=None):
    train = pd.read_csv(fnaddtr)
    test = pd.read_csv(fnaddte)
    train_index = np.unique(train['sid'].tolist())
    test_index = np.unique(test['sid'].tolist())
    train = train.drop(['sid','timeindex'],axis=1)
    test = test.drop(['sid','timeindex'],axis=1)
    if selected_features != None:
        train = train[selected_features]
        test = test[selected_features]
    tr = np.array(train)
    te = np.array(test)
    if len(train_index) != len(tr) or len(test_index) != len(te):
        exit('interesting.')

    pt_tr_add = {}
    for i in range(len(train_index)):
        pt_tr_add[train_index[i]] = i
    pt_te_add = {}
    for i in range(len(test_index)):
        pt_te_add[test_index[i]] = i

    tridlist = map(lambda x:pt_tr_add[int(x)],pt_tr)
    teidlist = map(lambda x:pt_te_add[int(x)],pt_te)

    tr = tr[tridlist]
    te = te[teidlist]

    # res_baseline = sklu.classify(tr, te, gt_tr, gt_te, LogisticRegression(penalty='l1', class_weight='balanced'), prob=True, norm=norm, header='%sdir'%(header))

    print data_tr.shape
    print data_te.shape
    mtr = np.hstack([data_tr,tr])
    mte = np.hstack([data_te,te])
    print mtr.shape
    print mte.shape

    print '# positive in test: %d' % sum(gt_te)
    if clf == None:
        clf = LogisticRegression(penalty='l2', class_weight='balanced')
    res = sklu.classify(mtr, mte, gt_tr, gt_te, clf, prob=True, norm=norm, header='%sdir'%(header))
    # return (res, res_baseline)
    return res

# '/data/mghcfl/mimic/ptsg_pca/pca_%d.pik'
def pcaClassify(data_tr, data_te, gt_tr, gt_te, nc=10, norm=None, fnfmt=None, clf=None, header="", fnroc=None):
    fn = fnfmt
    if os.path.isfile(fn):
        print('load %s' % (fn))
        with open(fn, 'rb') as f:
            [m, mtr, mte] = pickle.load(f)
    else:
        t1 = datetime.now()
        m = PCA(n_components=nc, whiten=True) # , tol=1e-5, max_iter=500
        m.fit(data_tr)
        mtr = m.transform(data_tr)
        mte = m.transform(data_te)

        t2 = datetime.now()
        td = t2 - t1
        print('PCA takes %d secs' % (td.seconds))

        with open(fn, 'wb') as f:
            pickle.dump([m, mtr, mte], f, -1)

    if clf == None:
        clf = LogisticRegression(penalty='l1', class_weight='auto')
    res = sklu.classify(mtr, mte, gt_tr, gt_te, clf, prob=True, norm=norm, header="%spca%d" % (header, nc))

    # print 'hello again'
    return (m, clf, res)