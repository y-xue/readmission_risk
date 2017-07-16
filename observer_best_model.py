import pandas as pd

import coding_util as cu
import sklearn_util as sklu
import plotting as pltg
import run
import pickle
import matplotlib.pyplot as plt
import os

import csv
from numpy import *
import random
from sklearn import metrics

import numpy as np
from scipy import interp

import math
import mimic_data as md

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, adjusted_mutual_info_score, roc_auc_score, mutual_info_score, make_scorer, roc_curve


# 0.636552954802
# 0.660662781705

# with LOS
# 0.641420481366 0.679273763143
# 0.662051201081 0.723652377094
def significant_test(oy,oa,ob,R):
	r = 0
	for i in range(R):
		X, Y = random_assign(oa,ob)
		if e_fun(oy,X,Y) >= e_fun(oy,oa,ob):
			r += 1
	return (r+1)/(R+1)

def random_assign(oa,ob):
	X, Y = [], []
	for i in range(len(oa)):
		la = oa[i]
		lb = ob[i]
		X.append([])
		Y.append([])
		for j in range(len(la)):
			if random.random() >= 0.5:
				X[i].append(la[j])
				Y[i].append(lb[j])
			else:
				X[i].append(lb[j])
				Y[i].append(la[j])
	return (X,Y)

def e_fun(oy,X,Y):
	return abs((get_mean_auc(oy,X) - get_mean_auc(oy,Y)))

def get_mean_auc(oy,res_list):
	'''
	Calculate mean AUC of all folds.
	'''
	mean_tpr = 0.0
	mean_fpr = linspace(0, 1)
	for i in range(5):
		fpr_te, tpr_te, thresholds_te = metrics.roc_curve(oy[i], res_list[i])
		mean_tpr += interp(mean_fpr, fpr_te, tpr_te)
		mean_tpr[0] = 0.0
	mean_tpr /= 5
	mean_tpr[-1] = 1.0
	mean_auc = metrics.auc(mean_fpr, mean_tpr)

	return mean_auc

def get_res_and_oc(input_folder, output_folder):
	seed = 2222
	standardize_method = 'z'
	is_cz = False
	freq_list = ['001','002','003','004','005','006','008','009', '01','011']
	freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}
	nel_graph_length = 13
	e = run.Experiment('%s/%s'%(input_folder,standardize_method),
		'%s/dataset'%(input_folder),
		seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	isg = 0
	freq_t = '011'
	# foldi = 0
	nc = 110
	c = 2
	pl = 'l1'
	cw = 'balanced'
	ntestth = 2
	
	best_features = ['urineByHrByWeight', 'HCT', 'INR', 'Platelets', 'RBC', 
	'DeliveredTidalVolume', 'PlateauPres', 'RAW', 'RSBI', 'mDBP', 'CV_HR', 
	'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_pH', 'Cl', 'Mg', 'Anticoagulant', 
	'beta.Blocking_agent', 'Somatostatin_preparation', 'Vasodilating_agent', 
	'AIDS', 'MetCarcinoma']

	# res_list = []
	# oc_list = []
	# oa = []
	# for foldi in range(5):
	# 	fnaddtr = '../../readmission_risk_baseline/data/seed2222/raw/interp/mean/last_measures/dataset/train_fold%d_%s.csv'%(foldi,standardize_method)
	# 	fnaddte = '../../readmission_risk_baseline/data/seed2222/raw/interp/mean/last_measures/dataset/test_fold%d_%s.csv'%(foldi,standardize_method)
	# 	prediction_matrics = e.read_prediction_matrics(isg,freq_t)
	# 	(res, gt_te, pt_te) = e.nmfClassify_ob(prediction_matrics['ptsg'][foldi],
	# 		prediction_matrics['ptwd'][foldi],
	# 		prediction_matrics['sgs'][foldi],
	# 		prediction_matrics['pt'][foldi],
	# 		prediction_matrics['gt'][foldi],
	# 		'%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(e.cdn,isg,freq_t,foldi,nc),
	# 		ntestth, foldi, nc, c, pl, cw, fnaddtr, fnaddte, best_features)
	# 	res_list.append(res)
	# 	oc_list.append(gt_te)
	# 	oa.append(res['n_pre_te'])
	# (auc, tr_auc) = e.get_mean_auc(res_list)
	# print auc, tr_auc

	res_list = []
	ob = []
	for foldi in range(5):
		prediction_matrics = e.read_prediction_matrics(isg,freq_t)
		res = e.nmfClassify(prediction_matrics['ptsg'][foldi],
			prediction_matrics['ptwd'][foldi],
			prediction_matrics['sgs'][foldi],
			prediction_matrics['pt'][foldi],
			prediction_matrics['gt'][foldi],
			'%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(e.cdn,isg,freq_t,foldi,nc),
			ntestth, foldi, nc, c, pl, cw)
		res_list.append(res)
		ob.append(res['n_pre_te'])
	(auc, tr_auc) = e.get_mean_auc(res_list)
	print auc, tr_auc

	# with open('%s/oy'%(output_folder),'wb') as f:
	# 	pickle.dump(oc_list,f)
	# with open('%s/oa'%(output_folder),'wb') as f:
	# 	pickle.dump(oa,f)
	# with open('%s/ob'%(output_folder),'wb') as f:
	# 	pickle.dump(ob,f)
	# return (oc_list, oa, ob)

# get_res_and_oc(
# 	'../data/seed2222/raw/interp/mean',
# 	'../observer/significant_test')
# oy = pickle.load(open('../observer/significant_test/oy','r'))
# oa = pickle.load(open('../observer/significant_test/oa','r'))
# ob = pickle.load(open('../observer/significant_test/ob','r'))
# print oy
# print oa
# print ob
# X,Y = random_assign(oa,ob)
# print oa[0][0:10]
# print ob[0][0:10]
# print X[0][0:10]
# print Y[0][0:10]
# print e_fun(oy,oa,ob)
# print significant_test(oy,oa,ob,1000)


def missing_percent(fin,fout):
	df = pd.read_csv(fin)
	gp = df.groupby('sid')
	fn = open(fout,'w')
	for sid, g in gp:
		(m,n) = g.shape
		missing_cnt = g.isnull().sum().sum()
		fn.write('%s: %d %d %.3f\n'%(sid,missing_cnt,m*n,missing_cnt*1.0/(m*n)))
	fn.close()

def test_time(input_folder):
	seed = 2222
	standardize_method = 'z'
	is_cz = False
	freq_list = ['001','002','003','004','005','006','008','009', '01','011']
	freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}
	nel_graph_length = 13
	
	e = run.Experiment('%s/%s'%(input_folder,standardize_method),
		'%s/dataset'%(input_folder),
		seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)
	
	for foldi in range(5):
		ftrnel = "%s/mimic_train_fold%d.nel"%(e.cdn,foldi)
		ftrnode = "%s/mimic_train_fold%d.node"%(e.cdn,foldi)
		fnel = "%s/mimic_fold%d.nel"%(e.cdn,foldi)
		fnode = "%s/mimic_fold%d.node"%(e.cdn,foldi)
		e.subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t='011', foldi=foldi)

def get_mean_auc(res_list):
	'''
	Calculate mean AUC of all folds.
	'''
	mean_tpr, mean_tpr_tr = 0.0, 0.0
	mean_fpr, mean_fpr_tr = np.linspace(0, 1), np.linspace(0, 1)
	for i in range(5):
		mean_tpr += interp(mean_fpr, res_list[i]['fpr_te'], res_list[i]['tpr_te'])
		mean_tpr_tr += interp(mean_fpr_tr, res_list[i]['fpr_tr'], res_list[i]['tpr_tr'])
		mean_tpr[0] = 0.0
		mean_tpr_tr[0] = 0.0
	mean_tpr /= 5
	mean_tpr_tr /= 5
	mean_tpr[-1] = 1.0
	mean_tpr_tr[-1] = 1.0
	mean_auc = metrics.auc(mean_fpr, mean_tpr)
	mean_auc_tr = metrics.auc(mean_fpr_tr, mean_tpr_tr)

	return (mean_fpr,mean_tpr,mean_auc)

def plot_auc():

	mean_fpr, mean_tpr, mean_auc = get_mean_auc(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/res_baseline_1170_list','r')))
	plt.plot(mean_fpr, mean_tpr, label='Baseline (area = %0.3f)' % mean_auc, lw=2)

	mean_fpr, mean_tpr, mean_auc = get_mean_auc(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/dir_res_list','r')))
	plt.plot(mean_fpr, mean_tpr, label='Subgraphs (area = %0.3f)' % mean_auc, lw=2)

	mean_fpr, mean_tpr, mean_auc = get_mean_auc(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/dir+baseline_res_list','r')))
	plt.plot(mean_fpr, mean_tpr, label='Subgraphs+snapshots (area = %0.3f)' % mean_auc, lw=2)

	mean_fpr, mean_tpr, mean_auc = get_mean_auc(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/nmf_res_list','r')))
	plt.plot(mean_fpr, mean_tpr, label='Grouping (area = %0.3f)' % mean_auc, lw=2)

	mean_fpr, mean_tpr, mean_auc = get_mean_auc(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/nmf+baseline_res_list','r')))
	plt.plot(mean_fpr, mean_tpr, label='Grouping+snapshots  (area = %0.3f)' % mean_auc, lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

def split_tr_te_by_ptids(gt, pt, fntr, fnte):
    i = 0
    hpttid = {}
    for p in pt:
        hpttid[p] = i
        i += 1

    train = pd.read_csv(fntr)
    test = pd.read_csv(fnte)

    train_index = np.unique(train['sid'].tolist())
    test_index = np.unique(test['sid'].tolist())

    filtered_tridlist = filter(lambda x: str(int(x)) in hpttid, train_index)
    filtered_teidlist = filter(lambda x: str(int(x)) in hpttid, test_index)

    tridlist = map(lambda x: hpttid[str(int(x))], filtered_tridlist)
    teidlist = map(lambda x: hpttid[str(int(x))], filtered_teidlist)

    gt1c = np.argmax(gt,1)
    pgt = np.array(gt1c).ravel()
    gt_tr = pgt[tridlist]
    gt_te = pgt[teidlist]

    # print pt
    # print tridlist

    pt_tr = pt[tridlist]
    pt_te = pt[teidlist]

    return (gt_tr, gt_te, pt_tr, pt_te)


def prf_tr_te(gt_te, pre_te, classes=None, verbose=1):
	print("*** Testing %d ***" % (len(pre_te)))
	maf1_te = f1_score(gt_te, pre_te, average='macro', pos_label=None)
	mif1_te = f1_score(gt_te, pre_te, average='micro', pos_label=None)
	marec_te = recall_score(gt_te, pre_te, average='macro', pos_label=None)
	mirec_te = recall_score(gt_te, pre_te, average='micro', pos_label=None)
	mapre_te = precision_score(gt_te, pre_te, average='macro', pos_label=None)
	mipre_te = precision_score(gt_te, pre_te, average='micro', pos_label=None)
	if verbose>=1:
		print(classification_report(gt_te, pre_te, target_names=classes))
		print(confusion_matrix(gt_te, pre_te))
	return (mapre_te, marec_te, maf1_te, mipre_te, mirec_te, mif1_te)

def clf(pre_te):
	for i in range(len(pre_te)):
		if pre_te[i] >= 0.5:
			pre_te[i] = 1
		else:
			pre_te[i] = 0
	return pre_te

def get_precision_recall(res_list):
	cdn = '../data/seed2222/raw/interp/mean'
	for foldi in range(5):
		ptsg = pickle.load(open('%s/z/isg0/pt_sg_w/ptsg_011_fold%d'%(cdn,foldi), 'rb'))
		ptwd = pickle.load(open('%s/z/isg0/pt_sg_w/ptwd_011_fold%d'%(cdn,foldi), 'rb'))
		sgs = pickle.load(open('%s/z/isg0/pt_sg_w/sgs_011_fold%d'%(cdn,foldi), 'rb'))
		gt = pickle.load(open('%s/z/isg0/pt_sg_w/gt_011_fold%d'%(cdn,foldi), 'rb'))
		pt = pickle.load(open('%s/z/isg0/pt_sg_w/pt_011_fold%d'%(cdn,foldi), 'rb'))
		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 2)
		(gt_tr, gt_te, pt_tr, pt_te) = split_tr_te_by_ptids(gt, pt,
			'%s/dataset/train_fold%d_z.csv'%(cdn,foldi),
			'%s/dataset/test_fold%d_z.csv'%(cdn,foldi))
		
		pre_te = res_list[foldi]['n_pre_te']
		pre_te = clf(pre_te)

		if foldi == 0:
			cte = gt_te
			c_pre_te = pre_te
			print pre_te
		else:
			cte = np.append(cte,gt_te)
			c_pre_te = np.append(c_pre_te,pre_te)

		print cte.shape
		print c_pre_te.shape
		
		# print pre_te
		# print pre_te.shape
		
		# cte = cte + gt_te
		# print cte
		# c_pre_te = c_pre_te + pre_te

	res = {}
	(res['mapre_te'], res['marec_te'], res['maf1_te'], res['mipre_te'], res['mirec_te'], res['mif1_te']) = prf_tr_te(cte, c_pre_te, classes=None, verbose=1)
	return res

def plot_precision_recall():
	if os.path.isfile('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/precision_list'):
		precision_list = pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/precision_list','r'))
		recall_list = pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/recall_list','r'))
		f1_list = pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/f1_list','r'))
	else:
		f1_list = []
		precision_list = []
		recall_list = []
		# x = ['BL', 'SG', 'SG+SNP', 'GROUPING', 'GROUPING+SNP']

		res = get_precision_recall(pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/res_baseline_1170_list','r')))
		# print res
		precision_list.append(res['mapre_te'])
		recall_list.append(res['marec_te'])
		f1_list.append(res['maf1_te'])

		# print precision_list
		# print recall_list
		# print res
		# plt.plot(, label='Baseline (precision = %0.3f)' % res['mapre_te'], lw=2)

		res = get_precision_recall(pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/dir_res_list','r')))
		# print res
		precision_list.append(res['mapre_te'])
		recall_list.append(res['marec_te'])
		f1_list.append(res['maf1_te'])

		# print precision_list
		# print recall_list
		# mean_fpr, mean_tpr, mean_auc = get_precision_recall(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/dir_res_list','r')))
		# plt.plot(mean_fpr, mean_tpr, label='Subgraphs (area = %0.3f)' % mean_auc, lw=2)

		res = get_precision_recall(pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/dir+baseline_res_list','r')))
		# print res
		precision_list.append(res['mapre_te'])
		recall_list.append(res['marec_te'])
		f1_list.append(res['maf1_te'])

		# print precision_list
		# print recall_list
		# mean_fpr, mean_tpr, mean_auc = get_precision_recall(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/dir+baseline_res_list','r')))
		# plt.plot(mean_fpr, mean_tpr, label='Subgraphs+snapshots (area = %0.3f)' % mean_auc, lw=2)

		res = get_precision_recall(pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/nmf_res_list','r')))
		# print res
		precision_list.append(res['mapre_te'])
		recall_list.append(res['marec_te'])
		f1_list.append(res['maf1_te'])

		# print precision_list
		# print recall_list
		# mean_fpr, mean_tpr, mean_auc = get_precision_recall(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/nmf_res_list','r')))
		# plt.plot(mean_fpr, mean_tpr, label='Grouping (area = %0.3f)' % mean_auc, lw=2)

		res = get_precision_recall(pickle.load(open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/nmf+baseline_res_list','r')))
		# print res
		precision_list.append(res['mapre_te'])
		recall_list.append(res['marec_te'])
		f1_list.append(res['maf1_te'])

		# print precision_list
		# print recall_list
		# print res
		# mean_fpr, mean_tpr, mean_auc = get_precision_recall(pickle.load(open('../observer/seed2222_raw_interp_mean_z_isg0_reslist/nmf+baseline_res_list','r')))
		# plt.plot(mean_fpr, mean_tpr, label='Grouping+snapshots  (area = %0.3f)' % mean_auc, lw=2)

		with open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/precision_list','w') as f:
			pickle.dump(precision_list,f)
		with open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/recall_list','w') as f:
			pickle.dump(recall_list,f)
		with open('/Users/XueY/Documents/research/readmission_risk/observer/seed2222_raw_interp_mean_z_isg0_reslist/f1_list','w') as f:
			pickle.dump(f1_list,f)

	x = [1,2,3,4,5]

	plt.plot(x, precision_list, label='precisions', lw=2)
	plt.plot(x, recall_list, label='recalls', lw=2)
	plt.plot(x, f1_list, label='f1', lw=2)
	plt.xlim([1, 5])
	plt.ylim([0.5, 0.65])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

# plot_precision_recall()
# plot_auc()

# {17795: (0, 0, 1), 11908: (1, 1, 0), ...}
# (true_label,temp_pre,base_pre)
def sid_pred_diff_btw_baseline_temporal():
	folder = '../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_s005'
	pre_te = pickle.load(open('%s/pre_te'%folder,'rb'))
	gt_te = pickle.load(open('%s/gt_te'%folder,'rb'))
	pt_te = pickle.load(open('%s/pt_te'%folder,'rb'))

	b_pre_te = pickle.load(open('%s/b_pre_te'%folder,'rb'))
	b_oc_te = pickle.load(open('%s/b_oc_te'%folder,'rb'))
	b_sid_te = pickle.load(open('%s/b_sid_te'%folder,'rb'))

	temp_dic = {}
	base_dic = {}

	for i in range(len(pre_te)):
		sid = int(pt_te[i])
		oc = gt_te[i]
		if pre_te[i] > 0.5:
			pre = 1
		else:
			pre = 0

		temp_dic[sid] = (oc,pre)

	for i in range(len(b_pre_te)):
		sid = int(b_sid_te[i])
		oc = b_oc_te[i]
		if b_pre_te[i] > 0.5:
			pre = 1
		else:
			pre = 0

		base_dic[sid] = (oc,pre)

	diff_dic = {}
	for k in temp_dic:
		if k in base_dic:
			if temp_dic[k][0] != base_dic[k][0]:
				exit('different outcome')
			oc = temp_dic[k][0]
			temp_pre = temp_dic[k][1]
			base_pre = base_dic[k][1]
			if temp_pre != base_pre:
				diff_dic[k] = (oc,temp_pre,base_pre)

	with open('../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_s005/diff_dic','wb') as f:
		pickle.dump(diff_dic,f)

def get_correct_wrong_clfd_pt(cdn):
	cl = []
	wl = []
	for foldi in range(5):
		ptsg = pickle.load(open('%s/z/isg0/pt_sg_w/ptsg_011_fold%d'%(cdn,foldi), 'rb'))
		ptwd = pickle.load(open('%s/z/isg0/pt_sg_w/ptwd_011_fold%d'%(cdn,foldi), 'rb'))
		sgs = pickle.load(open('%s/z/isg0/pt_sg_w/sgs_011_fold%d'%(cdn,foldi), 'rb'))
		gt = pickle.load(open('%s/z/isg0/pt_sg_w/gt_011_fold%d'%(cdn,foldi), 'rb'))
		pt = pickle.load(open('%s/z/isg0/pt_sg_w/pt_011_fold%d'%(cdn,foldi), 'rb'))
		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 2)
		(gt_tr, gt_te, pt_tr, pt_te) = split_tr_te_by_ptids(gt, pt,
			'%s/dataset/train_fold%d_z.csv'%(cdn,foldi),
			'%s/dataset/test_fold%d_z.csv'%(cdn,foldi))
		# pre_te = pickle.load(open('%s/%s/isg%d/res/c_pre_te_fold%d'%(cdn,standardize_method,isg,foldi)))
		pre_te = pickle.load(open('%s/z/isg0/res/c_pre_te_fold%d'%(cdn,foldi), 'rb'))

		for i in range(len(pt_te)):
			if pre_te[i] == gt_te[i]:
				cl.append(int(pt_te[i]))
			else:
				wl.append(int(pt_te[i]))
	
	return (cl,wl)

def get_los(cdn):
	tr = pd.read_csv('%s/dataset/train_fold0_z.csv'%(cdn))
	te = pd.read_csv('%s/dataset/test_fold0_z.csv'%(cdn))
	data = tr.append(te)

	dic = {}

	grouped = data.groupby('sid')
	lm = pd.DataFrame(columns=data.columns)

	for sid, group in grouped:
		t = group.sort(columns='timeindex',ascending = False)
		dic[int(sid)] = t['timeindex'].iloc[0]

	return dic

def partition_by_days_of_stay(los_dic):
	dic = {}
	for k in los_dic:
		d = math.floor(los_dic[k]/1440)
		if d not in dic:
			dic[d] = []
		dic[d].append(k)
	return dic

def count_pt(los_dic, correct_clfd_pts, wrong_clfd_pts):
	dic = {}
	for k in los_dic:
		dic[k] = {}
		dic[k]['correct'] = 0
		dic[k]['wrong'] = 0
		for p in los_dic[k]:
			if p in correct_clfd_pts:
				dic[k]['correct'] += 1
			elif p in wrong_clfd_pts:
				dic[k]['wrong'] += 1
			# else:
			# 	print('in count_pt: %s'%p)

	return dic

def los_experiments():
	cdn = '../data/seed2222/raw/interp/mean'
	(correct_clfd_pts, wrong_clfd_pts) = get_correct_wrong_clfd_pt(cdn)
	print len(correct_clfd_pts+wrong_clfd_pts)
	print len(set(correct_clfd_pts+wrong_clfd_pts))
	los_dic = get_los(cdn)
	los_dic = partition_by_days_of_stay(los_dic)
	los_count_dic = count_pt(los_dic, correct_clfd_pts, wrong_clfd_pts)

	print los_count_dic

# los_experiments()

# {0.0: {'wrong': 52, 'correct': 112}, 0.464
# 1.0: {'wrong': 103, 'correct': 199}, 0.518
# 2.0: {'wrong': 87, 'correct': 142},  0.613
# 3.0: {'wrong': 40, 'correct': 78},   0.513
# 4.0: {'wrong': 28, 'correct': 52},   0.538
# 5.0: {'wrong': 22, 'correct': 34},   0.647
# 6.0: {'wrong': 93, 'correct': 122},  0.762
# 7.0: {'wrong': 3, 'correct': 3}}     0.5
 
# >= 3
# 166 : 289

# >= 6
# 96 : 125 0.768

def get_wrongly_clfd_pt(low, high, foldi):
	badly_wronly_clfd_as_readmit_list = []
	badly_wronly_clfd_as_nonreadmit_list = []

	cdn = '../data/seed2222/raw/interp/mean'
	ptsg = pickle.load(open('%s/z/isg0/pt_sg_w/ptsg_011_fold%d'%(cdn,foldi), 'rb'))
	ptwd = pickle.load(open('%s/z/isg0/pt_sg_w/ptwd_011_fold%d'%(cdn,foldi), 'rb'))
	sgs = pickle.load(open('%s/z/isg0/pt_sg_w/sgs_011_fold%d'%(cdn,foldi), 'rb'))
	gt = pickle.load(open('%s/z/isg0/pt_sg_w/gt_011_fold%d'%(cdn,foldi), 'rb'))
	pt = pickle.load(open('%s/z/isg0/pt_sg_w/pt_011_fold%d'%(cdn,foldi), 'rb'))
	(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 2)
	(gt_tr, gt_te, pt_tr, pt_te) = split_tr_te_by_ptids(gt, pt,
		'%s/dataset/train_fold%d_z.csv'%(cdn,foldi),
		'%s/dataset/test_fold%d_z.csv'%(cdn,foldi))
	# pre_te = pickle.load(open('%s/%s/isg%d/res/c_pre_te_fold%d'%(cdn,standardize_method,isg,foldi)))
	# pre_te = pickle.load(open('%s/z/isg0/res/c_pre_te_fold%d'%(cdn,foldi), 'rb'))
	res = pickle.load(open('%s/z/isg0/res/res_fold%d'%(cdn,foldi), 'rb'))

	for i in range(len(pt_te)):
		if res['n_pre_te'][i] != gt_te[i]:
			if res['n_pre_te'][i] >= high and gt_te[i] == 0:
				badly_wronly_clfd_as_readmit_list.append(pt_te[i])
			if res['n_pre_te'][i] < low and gt_te[i] == 1:
				badly_wronly_clfd_as_nonreadmit_list.append(pt_te[i])

	print badly_wronly_clfd_as_readmit_list
	print badly_wronly_clfd_as_nonreadmit_list

	return badly_wronly_clfd_as_readmit_list, badly_wronly_clfd_as_nonreadmit_list

def get_ave_los_of_wrongly_clfd_pt(low, high):
	# pt_list = []
	readmit_ptlist = []
	nonreadmit_ptlist = []
	# total_clfd_readmit_num = 0
	# total_clfd_nonreadmit_num = 0
	for i in range(5):
		readmit_pts, nonreadmit_pts = get_wrongly_clfd_pt(low,high,i)
		readmit_ptlist += readmit_pts
		nonreadmit_ptlist += nonreadmit_pts
	
	los_dic = get_los('../data/seed2222/raw/interp/mean')

	readmit_los = 0
	for pt in readmit_ptlist:
		readmit_los += los_dic[int(pt)]

	nonreadmit_los = 0
	for pt in nonreadmit_ptlist:
		nonreadmit_los += los_dic[int(pt)]

	print len(readmit_ptlist)
	print readmit_los/len(readmit_ptlist)
	print len(nonreadmit_ptlist)
	print nonreadmit_los/len(nonreadmit_ptlist)
	print (readmit_los + nonreadmit_los) / len(readmit_ptlist+nonreadmit_ptlist)

# get_ave_los_of_wrongly_clfd_pt(0.1,0.9)

# <0.1,0.9>: 3 patients (2 clfd_readmit, 10046.5; 1, 10036), ave_los = 10043.0
# <0.2,0.8>: 17 patients (14 clfd_readmit, 5836.43; 3, 8149.0), ave_los = 6244.53
# <0.25,0.75>: 38 patients (27 clfd_readmit, 5530.41; 11, 5539.64), ave_los = 5533.08
# <0.3,0.7>: 65 patients (48 clfd_readmit, 5606.375; 17, 4895.0), ave_los = 5420.32
# <0.4,0.6>: 200 patients (139 clfd_readmit, 5198.45 ave_los; 61 clfd_nonreadmit, 4251.61), ave_los = 4909.66

# [138,19911,22383,25368,26401,6440,12920,13634,15486,20318,2322,5666,12467,7968,18123,20064,8686]

def get_ptsg_before_subgraph_filtering():
	cdn = '../data/seed2222/raw/interp/mean/z'
	isg = 0
	freq_t = '011'
	for foldi in range(5):
		fn_pt_sg_mat = "%s/isg%d/pt_sg_w/mimic_pt_sg.spmat_%s_fold%d"%(cdn,isg,freq_t,foldi)
		fn_pt_wd_mat = "%s/isg%d/pt_sg_w/mimic_pt_w.spmat_%s_fold%d"%(cdn,isg,freq_t,foldi)
		fn_sgs = "%s/isg%d/pt_sg_w/mimic.sgstr_%s_fold%d"%(cdn,isg,freq_t,foldi)
		fn_sparse_tensor = "%s/isg%d/pt_sg_w/mimic.tensor_fold%d"%(cdn,isg,foldi)
		fn_pt_gt = "%s/isg%d/pt_sg_w/mimic.ptmc_%s_fold%d" % (cdn,isg,freq_t,foldi)
		fn_pt_lab = "%s/isg%d/pt_sg_w/mimic.ptid_%s_fold%d"%(cdn,isg,freq_t,foldi)
		fn_sgtid = "%s/isg%d/pt_sg_w/mimic.sgtid_%s_fold%d"%(cdn,isg,freq_t,foldi)
		fn_ntid = "%s/isg%d/pt_sg_w/mimic.ntid_%s_fold%d"%(cdn,isg,freq_t,foldi)

		ptsg = md.read_pt_sg_mat(fn_pt_sg_mat)
		ptwd = md.read_pt_wd_mat(fn_pt_wd_mat)
		sgs = md.read_sgs(fn_sgs)
		spt = md.read_sparse_tensor(fn_sparse_tensor)

		print 'before_sg_filtering_sgs_fold%d: '%foldi, len(sgs)
		with open('%s/isg%d/pt_sg_w/before_sg_filtering_ptsg_%s_fold%d'%(cdn,isg,freq_t,foldi),'wb') as f:
			pickle.dump(ptsg,f)
		with open('%s/isg%d/pt_sg_w/before_sg_filtering_sgs_%s_fold%d'%(cdn,isg,freq_t,foldi),'wb') as f:
			pickle.dump(sgs,f)

		(hiso, hsgstr, hsgc, hsgsize) = md.sg_subiso(sgs)
		(ptsg, sgs, sptsel) = md.filter_sg(ptsg, hiso, sgs, spt=spt)
		# (ptsg, sgs) = md.filter_sg(ptsg, hiso, sgs)
		print 'after_sg_filtering_sgs_fold%d: '%foldi, len(sgs)

# get_ptsg_before_subgraph_filtering()

def get_ave_sg_for_each_patient():
	pre_ave_sg = 0
	ave_sg = 0
	for i in range(5):
		pre_ptsg = pickle.load(open('../data/seed2222/raw/interp/mean/z/isg0/pt_sg_w/before_sg_filtering_ptsg_011_fold0','r'))
		pre_ave_sg += (pre_ptsg.sum() / len(pre_ptsg))

		ptsg = pickle.load(open('../data/seed2222/raw/interp/mean/z/isg0/pt_sg_w/ptsg_011_fold0','r'))
		ave_sg += (ptsg.sum() / len(ptsg))

	print pre_ave_sg / 5
	print ave_sg / 5

get_ave_sg_for_each_patient()