""" Modified by yxue - 05-18-2017
First created by yluo - 07/29/2014
"""

import numpy as np
import os
import csv
import subprocess
import scipy as sp
import pickle
from sklearn import mixture
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, adjusted_mutual_info_score, roc_auc_score, mutual_info_score, make_scorer, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer, scale

def col_l1_norm(mdata):
    col_mean = np.nanmean(mdata, axis=0)
    mdata = mdata - col_mean[np.newaxis, :]
    col_max = np.nanmax(mdata, axis=0) # nan
    col_max[col_max == 0] = 1.
    mdata = mdata / col_max[np.newaxis, :] # using broadcasting
    return mdata;

def mean_imputing(mdata):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(mdata)
    return imp.transform(mdata);

def prf_cv(clf, data, target, cv=5):
    mapre_scorer = make_scorer(precision_score, average='macro', pos_label=None)
    marec_scorer = make_scorer(recall_score, average='macro', pos_label=None)
    maf1_scorer = make_scorer(f1_score, average='macro', pos_label=None)
    
    mipre_scorer = make_scorer(precision_score, average='micro', pos_label=None)
    mirec_scorer = make_scorer(recall_score, average='micro', pos_label=None)
    mif1_scorer = make_scorer(f1_score, average='micro', pos_label=None)

    mapre_cv = model_selection.cross_val_score(clf, data, target, cv=5, scoring=mapre_scorer)
    marec_cv = model_selection.cross_val_score(clf, data, target, cv=5, scoring=marec_scorer)
    maf1_cv = model_selection.cross_val_score(clf, data, target, cv=5, scoring=maf1_scorer)

    mipre_cv = model_selection.cross_val_score(clf, data, target, cv=5, scoring=mipre_scorer)
    mirec_cv = model_selection.cross_val_score(clf, data, target, cv=5, scoring=mirec_scorer)
    mif1_cv = model_selection.cross_val_score(clf, data, target, cv=5, scoring=mif1_scorer)

    print('%d-fold cv mapre: %s' % (cv, mapre_cv))
    print('%d-fold cv marec: %s' % (cv, marec_cv))
    print('%d-fold cv maf1: %s' % (cv, maf1_cv))

    print('%d-fold cv mipre: %s' % (cv, mipre_cv))
    print('%d-fold cv mirec: %s' % (cv, mirec_cv))
    print('%d-fold cv mif1: %s' % (cv, mif1_cv))
    return (mapre_cv, marec_cv, maf1_cv, mipre_cv, mirec_cv, mif1_cv);

def prf_tr_te(gt_tr, gt_te, pre_tr, pre_te, classes=None, verbose=1):
    if verbose>=1:
        print("*** Training %d ***" % (len(pre_tr)))
    maf1_tr = f1_score(gt_tr, pre_tr, average='macro', pos_label=None)
    mif1_tr = f1_score(gt_tr, pre_tr, average='micro', pos_label=None)

    marec_tr = recall_score(gt_tr, pre_tr, average='macro', pos_label=None)
    mirec_tr = recall_score(gt_tr, pre_tr, average='micro', pos_label=None)

    mapre_tr = precision_score(gt_tr, pre_tr, average='macro', pos_label=None)
    mipre_tr = precision_score(gt_tr, pre_tr, average='micro', pos_label=None)
    if verbose>=1:
        print(classification_report(gt_tr, pre_tr, target_names=classes))
        print(confusion_matrix(gt_tr, pre_tr))
        
        print("*** Testing %d ***" % (len(pre_te)))
    # print gt_te.shape
    # print pre_te.shape
    maf1_te = f1_score(gt_te, pre_te, average='macro', pos_label=None)
    mif1_te = f1_score(gt_te, pre_te, average='micro', pos_label=None)
    marec_te = recall_score(gt_te, pre_te, average='macro', pos_label=None)
    mirec_te = recall_score(gt_te, pre_te, average='micro', pos_label=None)
    mapre_te = precision_score(gt_te, pre_te, average='macro', pos_label=None)
    mipre_te = precision_score(gt_te, pre_te, average='micro', pos_label=None)
    if verbose>=1:
        print(classification_report(gt_te, pre_te, target_names=classes))
        print(confusion_matrix(gt_te, pre_te))
    return (mapre_tr, marec_tr, maf1_tr, mapre_te, marec_te, maf1_te, mipre_tr, mirec_tr, mif1_tr, mipre_te, mirec_te, mif1_te);

def auc_tr_te(gt_tr, gt_te, prob_pre_tr, prob_pre_te):
    # for i in range(len(nfer_pre_tr)):
    #     if cfer_pre_tr[i] == 0:
    #         nfer_pre_tr[i] = 1 - nfer_pre_tr[i]

    # for i in range(len(nfer_pre_te)):
    #     if cfer_pre_te[i] == 0:
    #         nfer_pre_te[i] = 1 - nfer_pre_te[i]

    auc_tr = roc_auc_score(gt_tr, prob_pre_tr)
    auc_te = roc_auc_score(gt_te, prob_pre_te)
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(gt_tr, prob_pre_tr)
    fpr_te, tpr_te, thresholds_te = roc_curve(gt_te, prob_pre_te)
    return (auc_tr, fpr_tr, tpr_tr, auc_te, fpr_te, tpr_te);

def classify(mtri, mtei, ctr, cte, clf, verbose=1, prob=False, norm=None, classes=None, header=""): 
    """ I designated the classifier specification to the user
    , kernel='linear', clfn='svm', cweight='auto', C=1.0
    examples as follow:
    if clfn == 'svm':
        clf = SVC(kernel=kernel, probability=prob, class_weight=cweight)
    elif clfn == 'lr':
        clf = LogisticRegression(penalty='l1', class_weight=cweight)
    elif clfn == "knn":
        clf = KNeighborsClassifier()
    else:
        print('unknown classifier %s' % (clfn))
        return;
    """

    t1 = datetime.now()
    res = {}
    if norm != None:
        mtr = normalize(mtri, norm=norm, copy=True)
        mte = normalize(mtei, norm=norm, copy=True)
    else:
        mtr = mtri
        mte = mtei
        # mte = col_l1_norm(np.asarray(mte.todense()))
    if verbose>1:
        prf_cv(clf, mtr, ctr)
    
    clf.fit(mtr, ctr)
    if verbose>=1:
        print('model trained')
    # The mean square error
    c_pre_te = clf.predict(mte)
    c_pre_tr = clf.predict(mtr)
    res['c_pre_tr'] = c_pre_tr
    res['c_pre_te'] = c_pre_te
    if prob:
        c_prob_tr = clf.predict_proba(mtr)
        c_prob_te = clf.predict_proba(mte)
        if verbose>=2:
            print("training data prob:")
            for i in range(len(c_pre_tr)):
                print("%s: %s" % (c_pre_tr[i], np.max(c_prob_tr[i])))
            print("testing data prob:")
            for i in range(len(c_pre_te)):
                print("%s: %s" % (c_pre_te[i], np.max(c_prob_te[i])))
        n_pre_tr = c_prob_tr[:,1]; res['n_pre_tr'] = n_pre_tr
        n_pre_te = c_prob_te[:,1]; res['n_pre_te'] = n_pre_te
    
    (res['mapre_tr'], res['marec_tr'], res['maf1_tr'], res['mapre_te'], res['marec_te'], res['maf1_te'], res['mipre_tr'], res['mirec_tr'], res['mif1_tr'], res['mipre_te'], res['mirec_te'], res['mif1_te']) = prf_tr_te(ctr, cte, c_pre_tr, c_pre_te, classes=classes, verbose=verbose)
    if prob:
        (res['auc_tr'], res['fpr_tr'], res['tpr_tr'], res['auc_te'], res['fpr_te'], res['tpr_te']) = auc_tr_te(ctr, cte, n_pre_tr, n_pre_te)
    t2 = datetime.now()
    td = t2 - t1
    # if verbose>=1:
    #     print('classification takes %f secs' % (td.total_seconds()))
    
    if verbose>=1:
        print('%s   \tpre tr\trec tr\tf1 tr\tpre te\trec te\tf1 te' % (header))
        print("%s ma\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (header, res['mapre_tr'], res['marec_tr'], res['maf1_tr'], res['mapre_te'], res['marec_te'], res['maf1_te']))
        print("%s mi\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (header, res['mipre_tr'], res['mirec_tr'], res['mif1_tr'], res['mipre_te'], res['mirec_te'], res['mif1_te']))
        if prob:
            print("%s auc\ttr=%.3f\tte=%.3f" % (header, res['auc_tr'], res['auc_te']))
            
    return res;
    
