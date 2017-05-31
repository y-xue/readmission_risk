import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.decomposition import PCA

import pickle
# import matplotlib.pyplot as plt
import os

from time import sleep

from rfe import *


from sklearn import mixture
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, adjusted_mutual_info_score, roc_auc_score, mutual_info_score, make_scorer, roc_curve
from sklearn import model_selection
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer, scale


def lr(cdn, feature_type, c=1, penal=None, bl=None, rfe_n=None):
	print 'lr...'
	clf_type = 'lr'
	
	thres = [0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]

	auc_d, acc_d, f1_d = {}, {}, {}
	auc_d['bauc'], auc_d['bf'], acc_d['ba'], f1_d['bf'] = 0, 0, 0, 0

	# print 'c = %s'%str(c)

	clf = LogisticRegression(penalty=penal, C=c, class_weight=bl)
	dif_param = penal

	if rfe_n != None:
		rfe(cdn,rfe_n,feature_type,c,penal,bl)
		rfe_cols = pickle.load(open('%s/rfe_cols/%s/best%d_c%s_%s_%s'%(cdn,feature_type,rfe_n,str(c),penal,bl),'rb'))
		# print 'rfe_cols/%s/best%d_c%s_%s_%s'%(feature_type,rfe_n,str(c),penal,bl)

	predicted = {}
	proba = np.array([])
	y_label = np.array([])

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1)

	for k in range(5):
		# train = pd.read_csv('../data/lm_after_mice/train_mice_z/last12h_pmm_mp%s_mc%s_fold%d.csv'%(minp,minc,k))
		if feature_type is None:
			train  = pd.read_csv('%s/dataset/train_fold%d.csv'%(cdn,k))
			test = pd.read_csv('%s/dataset/test_fold%d.csv'%(cdn,k))
		else:
			train = pd.read_csv('%s/dataset/train_fold%d_%s.csv'%(cdn,k,feature_type))
			test = pd.read_csv('%s/dataset/test_fold%d_%s.csv'%(cdn,k,feature_type))
		x_train = train.drop(['readmit','sid'],axis=1)
		y_train = train['readmit']
		x_test = test.drop(['readmit','sid'],axis=1)
		y_test = test['readmit']

		x_train = x_train[rfe_cols]
		x_test = x_test[rfe_cols]
		# print x_train.shape

		clf = clf.fit(x_train, y_train)
		
		y_prob = clf.predict_proba(x_test)
		proba = np.append(proba, y_prob[:,1])
		y_label = np.append(y_label, y_test)

		fpr_te, tpr_te, thresholds_te = metrics.roc_curve(y_test, y_prob[:,1])
		mean_tpr += np.interp(mean_fpr, fpr_te, tpr_te)

		for t in thres:
			if t not in predicted:
				predicted[t] = np.array([])

			y_pred = np.array([])
			for p in y_prob:
				if p[1] > t:
					y_pred = np.append(y_pred, 1)
				else:
					y_pred = np.append(y_pred, 0)
			predicted[t] = np.append(predicted[t], y_pred)

	mean_tpr[0] = 0.0
	mean_tpr /= 5
	mean_tpr[-1] = 1.0
	mean_auc = metrics.auc(mean_fpr, mean_tpr)
	
	auc_roc = metrics.roc_auc_score(y_label, proba)
	if mean_auc != auc_roc:
		# print 'mean_auc:', mean_auc
		# print 'auc_roc:', auc_roc
		auc_roc = mean_auc
	
	for t in thres:
		acc = metrics.accuracy_score(y_label, predicted[t])
		precision = metrics.precision_score(y_label, predicted[t])
		recall = metrics.recall_score(y_label, predicted[t])
		f1 = metrics.f1_score(y_label, predicted[t])

		if f1 > f1_d['bf']:
			f1_d['ba'], f1_d['bp'], f1_d['br'], f1_d['bf'], f1_d['bauc'] = acc, precision, recall, f1, auc_roc
			f1_d['bc'], f1_d['bt'] = c, t

		if auc_roc > auc_d['bauc'] or (auc_roc == auc_d['bauc'] and f1 > auc_d['bf']):
			auc_d['ba'], auc_d['bp'], auc_d['br'], auc_d['bf'], auc_d['bauc'] = acc, precision, recall, f1, auc_roc
			auc_d['bc'], auc_d['bt'] = c, t

		if acc > acc_d['ba'] and precision != 0:
			acc_d['ba'], acc_d['bp'], acc_d['br'], acc_d['bf'], acc_d['bauc'] = acc, precision, recall, f1, auc_roc
			acc_d['bc'], acc_d['bt'] = c, t


	output = ''
	output += 'tune auc:\n'
	output += 'accuracy_score = %f\n' % auc_d['ba']
	output += 'precision_score = %f\n' % auc_d['bp']
	output += 'recall_score = %f\n' % auc_d['br']
	output += 'f1_score = %f\n' % auc_d['bf']
	output += 'auc_roc = %f\n' % auc_d['bauc']
	output += 'best c = %f\n' % auc_d['bc']
	output += 'best threshold = %f\n' % auc_d['bt']

	output += '\ntune f1:\n'
	output += 'accuracy_score = %f\n' % f1_d['ba']
	output += 'precision_score = %f\n' % f1_d['bp']
	output += 'recall_score = %f\n' % f1_d['br']
	output += 'f1_score = %f\n' % f1_d['bf']
	output += 'auc_roc = %f\n' % f1_d['bauc']
	output += 'best c = %f\n' % f1_d['bc']
	output += 'best threshold = %f\n' % f1_d['bt']

	output += '\ntune acc:\n'
	output += 'accuracy_score = %f\n' % acc_d['ba']
	output += 'precision_score = %f\n' % acc_d['bp']
	output += 'recall_score = %f\n' % acc_d['br']
	output += 'f1_score = %f\n' % acc_d['bf']
	output += 'auc_roc = %f\n' % acc_d['bauc']
	output += 'best c = %f\n' % acc_d['bc']
	output += 'best threshold = %f\n' % acc_d['bt']

	# if not os.path.exists('output/%s'%nth_imp):
	# 	os.makedirs('output/%s'%nth_imp)


	if rfe_n == None:
		checkAndCreate("%s/%soutput/pca/%s"%(cdn,clf_type+'_',feature_type))
		ofile = open("%s/%soutput/pca/%s/%s_%s_pca%d.txt" % (cdn,clf_type+'_',feature_type, dif_param, bl, pca_n), 'w')
	else:
		checkAndCreate('%s/%soutput/rfe/%s'%(cdn,clf_type+'_',feature_type))
		ofile = open("%s/%soutput/rfe/%s/%s_%s_rfe%d.txt" % (cdn,clf_type+'_',feature_type, dif_param, bl, rfe_n), 'w')
	
	ofile.write(output)
	ofile.close()

	return auc_d['bauc']


def lr_ob(ea_cdn, feature_type, k=0, c=1.0, penal=None, kl=None, mf=None, bl=None, imp='mean', minp="0", minc="0", clf_type='lr', rfe_n=None, pca_n=None, add_mean_vals=False, rfe_with_mn=False):
	rfe_cols = pickle.load(open('%s/rfe_cols/%s/best%d_mp%s_mc%s_c%s_%s_%s'%(ea_cdn,feature_type,rfe_n,minp,minc,str(c),penal,bl),'rb'))
				
	train = pd.read_csv('%s/dataset/train_fold%d_%s.csv'%(ea_cdn,k,feature_type))
	test = pd.read_csv('%s/dataset/test_fold%d_%s.csv'%(ea_cdn,k,feature_type))
	x_train = train[rfe_cols]
	x_test = test[rfe_cols]
	y_train = train['readmit']
	y_test = test['readmit']
	te_sidlist = test['sid'].tolist()

	y_label = np.array(y_test)

	clf = LogisticRegression(penalty=penal, C=c, class_weight=bl)
	res = classify(x_train, x_test, y_train, y_test, clf, prob=True)

	return res
	# clf = clf.fit(x_train, y_train)
	# y_prob = clf.predict_proba(x_test)
	# print metrics.roc_auc_score(y_label, y_prob[:,1])
	# checkAndCreate('../observer/error_analysis/seed2222_raw_interp_mean_z')
	# with open('../observer/error_analysis/seed2222_raw_interp_mean_z/pre_te','wb') as f:
	# 	pickle.dump(y_prob[:,1],f)
	# with open('../observer/error_analysis/seed2222_raw_interp_mean_z/oc_te','wb') as f:
	# 	pickle.dump(y_label,f)
	# with open('../observer/error_analysis/seed2222_raw_interp_mean_z/sid_te','wb') as f:
	# 	pickle.dump(te_sidlist,f)

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

    # t1 = datetime.now()
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
    # t2 = datetime.now()
    # td = t2 - t1
    # if verbose>=1:
    #     print('classification takes %f secs' % (td.total_seconds()))
    
    if verbose>=1:
        print('%s   \tpre tr\trec tr\tf1 tr\tpre te\trec te\tf1 te' % (header))
        print("%s ma\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (header, res['mapre_tr'], res['marec_tr'], res['maf1_tr'], res['mapre_te'], res['marec_te'], res['maf1_te']))
        print("%s mi\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (header, res['mipre_tr'], res['mirec_tr'], res['mif1_tr'], res['mipre_te'], res['mirec_te'], res['mif1_te']))
        if prob:
            print("%s auc\ttr=%.3f\tte=%.3f" % (header, res['auc_tr'], res['auc_te']))
            
    return res;
    