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


def check_nmfClassify(input_folder, output_folder, isg, freq_t, nc, c, pl, cw, ntestth):
	seed = 2222
	standardize_method = 'z'
	is_cz = False
	freq_list = ['001','002','003','004','005','006','008','009', '01','011']
	freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}
	nel_graph_length = 13
	e = run.Experiment('%s/%s'%(input_folder,standardize_method),
		'%s/dataset'%(input_folder),
		seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)


	res_list = []
	for foldi in range(5):
		prediction_matrics = e.read_prediction_matrics(isg,freq_t,cfolder='%s/isg%d/same_freq_t/pt_sg_w'%(e.cdn,isg))
		res = e.nmfClassify(prediction_matrics['ptsg'][foldi],
			prediction_matrics['ptwd'][foldi],
			prediction_matrics['sgs'][foldi],
			prediction_matrics['pt'][foldi],
			prediction_matrics['gt'][foldi],
			'%s/isg%d/nmf_piks/nmf_%s_fold%d_%d_ramdon0-again.pik'%(output_folder,isg,freq_t,foldi,nc),
			ntestth, foldi, nc, c, pl, cw)
		res_list.append(res)
	(auc, tr_auc) = e.get_mean_auc(res_list)
	print auc, tr_auc
	# isg 1: freq_t 001: 0.601825 (isg1,s001,nc100,c2,l1,balanced)

# check_nmfClassify(
# 	'../data/seed2222/raw/interp/mean',
# 	'../observer/error_analysis/interp_mean_z_nmfClassify',
# 	1, '001', 100, 2, 'l1', 'balanced', 2)
# 0.591315152905 0.687114670838
# len(sgs): 1421
# len(sgs): 1410
# len(sgs): 1431
# len(sgs): 1408
# exclude pt 16748
# len(sgs): 1393
# exclude pt 16748
# 0.583946944709 0.681314517091
# random0: 0.599465558984 0.690203845117
# random0-again 0.599465558984 0.690187583632

# check_nmfClassify(
# 	'../data/seed2222/raw/interp/mean',
# 	'../observer/error_analysis/interp_mean_z_nmfClassify',
# 	1, '002', 100, 2, 'l1', 'balanced', 2)
# 0.602701703653 0.688779841628
# len(sgs): 1421
# len(sgs): 1410
# len(sgs): 1431
# len(sgs): 1408
# exclude pt 16748
# len(sgs): 1393
# exclude pt 16748
# 0.599377512716 0.688931335158
# random0: 0.599465558984 0.690187583632
# random0-again: 0.599465558984 0.690187583632


def missing_percent(fin,fout):
	df = pd.read_csv(fin)
	gp = df.groupby('sid')
	fn = open(fout,'w')
	for sid, g in gp:
		(m,n) = g.shape
		missing_cnt = g.isnull().sum().sum()
		fn.write('%s: %d %d %.3f\n'%(sid,missing_cnt,m*n,missing_cnt*1.0/(m*n)))
	fn.close()

def plot_imputes_values(l,output_folder,istrain=False):
	# if istrain:
	# 	pltg.plot_ori_vs_imp('../data/seed2222/raw/train_fold0.csv', 
	# 		'../data/seed2222/raw/mice/mp0.5_mc0.6/dataset/train_fold0.csv', 
	# 		'../observer/plot_imputes_values_new/seed2222_raw_mice_mp0.5mc0.6/fold0/train',
	# 		byPt=True, patient_idlist=l)
	# else:
	# 	pltg.plot_ori_vs_imp('../data/seed2222/raw/test_fold0.csv', 
	# 		'../data/seed2222/raw/mice/mp0.5_mc0.6/dataset/test_fold0.csv', 
	# 		'../observer/plot_imputes_values_new/seed2222_raw_mice_mp0.5mc0.6/fold0/test',
	# 		byPt=True, patient_idlist=l)
	for i in range(5):
		# df = pd.read_csv('../data/seed2222/raw/interp/mean/dataset/test_fold%d.csv'%i)
		# li = []
		# # print type(df['sid'][0]),df['sid'][0]
		# for pt in l:
		# 	# print pt
		# 	if float(pt) in df['sid'].tolist():
		# 		li.append(pt)
		# print li
		tr = pd.read_csv('../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i)
		mn = tr.mean()
		sd = tr.std()
		if istrain:
			pltg.plot_ori_vs_imp('../data/seed2222/raw/train_fold%d.csv'%i, 
				'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i, 
				'~/Desktop',
				# '../observer/plot_imputes_values_interp/seed2222_raw_mice_mp0.5mc0.6/fold%d/train'%i,
				byPt=True, patient_idlist=l, mn=mn,sd=sd)
		else:
			pltg.plot_ori_vs_imp('../data/seed2222/raw/test_fold%d.csv'%i, 
				'../data/seed2222/raw/interp/mean/dataset/test_fold%d.csv'%i, 
				'%s/fold%d'%(output_folder,i),
				byPt=True, patient_idlist=li,
				mn=mn,sd=sd)
	
	# if istrain:
	# 	pltg.plot_ori_vs_imp('../data/seed2222_one_hour_interval/raw/train_fold2.csv', 
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/train_fold2.csv', 
	# 		'../observer/plot_imputes_values_new/seed2222_one_hour_interval_raw_mice_mp0.5mc0.6/train',
	# 		byPt=True, patient_idlist=l)
	# else:
	# 	pltg.plot_ori_vs_imp('../data/seed2222_one_hour_interval/raw/test_fold2.csv', 
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/test_fold2.csv', 
	# 		'../observer/plot_imputes_values_new/seed2222_one_hour_interval_raw_mice_mp0.5mc0.6/test',
	# 		byPt=True, patient_idlist=l)

def check_interpolation_and_subgraphs():
	ft = 'raw'
	minp = 0.5
	minc = 0.6
	seed = 2222

	standardize_method = "cz"
	is_cz = True
	# standardize_method = "z"
	# is_cz = False

	freq_list = ['001','002','003','004','005','006','008','009', '01','011']
	freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}
	nel_graph_length = 13

	fout = '../observer/check_interpolation_and_subgraphs/seed%s_%s_mice_mp%s_mc%s_%s'%(seed,ft,minp,minc,standardize_method)
	cu.checkAndCreate(fout)

	cu.checkAndCreate('../data/seed%s/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method))
	e = run.Experiment('../data/seed%s/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method),
		'../data/seed%s/%s/mice/mp%s_mc%s/dataset'%(seed,ft,minp,minc),
		seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)


	foldi = 2

	train = e.ftrain%(e.dataset_folder,foldi,e.standardize_method)
	test = e.ftest%(e.dataset_folder,foldi,e.standardize_method)

	print train
	print test

	ftrnel = "%s/mimic_train_fold%d.nel"%(fout,foldi)
	ftrnode = "%s/mimic_train_fold%d.node"%(fout,foldi)
	fnel = "%s/mimic_fold%d.nel"%(fout,foldi)
	fnode = "%s/mimic_fold%d.node"%(fout,foldi)

	# e.interpolation(trcsv=train, tecsv=test, ftrnel=ftrnel, ftrnode=ftrnode, fnel=fnel, fnode=fnode)
	e.subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t='011', foldi=foldi, cfolder=fout)

# missing_percent('../data/seed2222/raw/train_fold2.csv','../observer/missing_percent.txt')
# missing_percent('../data/seed2222_one_hour_interval/raw/train_fold2.csv','../observer/missing_percent_seed2222_one_hour_interval_raw_train_fold2.txt')
# missing_percent('../data/seed2222_one_hour_interval/raw/test_fold2.csv','../observer/missing_percent_seed2222_one_hour_interval_raw_test_fold2.txt')
# l = [61,184,12940]
# l = [5362,5620]
# l = [3860,3700]
# l = [61]

# l = [3602,17122,18094,2333,5581,18022,6212,18420,17795,19334,22579,5278]
# l = [17]
# plot_imputes_values(l)

# l = [19148,22332,5450,18677,22083]
# plot_imputes_values(l,istrain=True)
# l = [1485,4316,9887]
# plot_imputes_values(l,istrain=False)

# l = [19148,22332,5450,18677,22083]
# plot_imputes_values(l,istrain=True)
# l = [1485,4316,9887]
# plot_imputes_values(l,istrain=False)

# l = [130,323,14824,20187,25317]
# plot_imputes_values(l,istrain=True)
# l = [21,1840,4271,4749,5908,22553,24958,25384]
# plot_imputes_values(l,istrain=False)

# check_interpolation_and_subgraphs()

# seed2222_one_hour_interval_raw_train_fold2:
# 19148.0: 94 2184 0.043
# 22332.0: 53 78 0.679
# 22083.0: 483 936 0.516
# 5450.0: 770 2964 0.260
# 18677.0: 3236 12012 0.269

# seed2222_one_hour_interval_raw_test_fold2:
# 1485.0: 1224 5460 0.224
# 4316.0: 723 1638 0.441
# 9887.0: 1456 3510 0.415

# error analysis

 
# 
# ft = 'raw'
# minp = 0.5
# minc = 0.6
# seed = 2222
# standardize_method = 'z'
# 
# Temporal
# 
# isg = 0
# freq_t = '006'
# foldi = 0
# nc = 90
# c = 1
# pl = 'l1'
# cw = 'balanced'
# ntestth = 2
# 
# 0.644425987449

# Baseline
# 
# pl = 'l1'
# bl = 'balanced'
# c = 0.2
# foldi = 0
# ft = 'z'
# r = 40
# 
# 0.584615384615

# freq_t 005: 0.629625 (isg0,s005,nc80,c1,l1,balanced)

# 0.636553, params: isg5,s011,nc110,c2,l1,balanced
# (dirClassify) freq_t 004: 0.553539 (isg0,s004,nc10,c1,l1,balanced)
def error_analysis(input_folder, output_folder):
	ft = 'raw'
	minp = 0.5
	minc = 0.6
	seed = 2222
	standardize_method = 'z'
	is_cz = False
	freq_list = ['001','002','003','004','005','006','008','009', '01','011']
	freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}
	nel_graph_length = 13
	# cu.checkAndCreate('../data/seed%s/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method))
	e = run.Experiment('%s/%s'%(input_folder,standardize_method),
		'%s/dataset'%(input_folder),
		seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# e = run.Experiment('../data/seed%s/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method),
	# 	'../data/seed%s/%s/mice/mp%s_mc%s/dataset'%(seed,ft,minp,minc),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# NMF:
	isg = 0
	freq_t = '011'
	nc = 110
	c = 2
	pl = 'l1'
	cw = 'balanced'
	ntestth = 2

	# DirClassify:
	# isg = 0
	# freq_t = '004'
	# c = 1
	# pl = 'l1'
	# cw = 'balanced'
	# ntestth = 2
	
	best_features = ['urineByHrByWeight', 'HCT', 'INR', 'Platelets', 'RBC', 
	'DeliveredTidalVolume', 'PlateauPres', 'RAW', 'RSBI', 'mDBP', 'CV_HR', 
	'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_pH', 'Cl', 'Mg', 'Anticoagulant', 
	'beta.Blocking_agent', 'Somatostatin_preparation', 'Vasodilating_agent', 
	'AIDS', 'MetCarcinoma']

	res_list = []
	# res_baseline_list = []
	for foldi in range(5):
		fnaddtr = '../../readmission_risk_baseline/data/seed2222/raw/interp/mean/last_measures/dataset/train_fold%d_%s_t.csv'%(foldi,standardize_method)
		fnaddte = '../../readmission_risk_baseline/data/seed2222/raw/interp/mean/last_measures/dataset/test_fold%d_%s_t.csv'%(foldi,standardize_method)
		prediction_matrics = e.read_prediction_matrics(isg,freq_t)
		(res, gt_te, pt_te, res_baseline) = e.nmfClassify_ob(prediction_matrics['ptsg'][foldi],
		# res = e.nmfClassify(prediction_matrics['ptsg'][foldi],
			prediction_matrics['ptwd'][foldi],
			prediction_matrics['sgs'][foldi],
			prediction_matrics['pt'][foldi],
			prediction_matrics['gt'][foldi],
			'%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(e.cdn,isg,freq_t,foldi,nc),
			ntestth, foldi, nc, c, pl, cw, fnaddtr, fnaddte, best_features)
		
		# res = e.dirClassify(prediction_matrics['ptsg'][foldi],
		# # res = e.nmfClassify(prediction_matrics['ptsg'][foldi],
		# 	prediction_matrics['ptwd'][foldi],
		# 	prediction_matrics['sgs'][foldi],
		# 	prediction_matrics['pt'][foldi],
		# 	prediction_matrics['gt'][foldi],
		# 	ntestth, foldi, c, pl, cw)
		res_list.append(res)
		# res_baseline_list.append(res_baseline)
	# with open('%s/res_baseline_1170_list'%(output_folder),'wb') as f:
	# 	pickle.dump(res_baseline_list,f)
	(auc, tr_auc) = e.get_mean_auc(res_list)
	print auc, tr_auc
	# (auc, tr_auc) = e.get_mean_auc(res_baseline_list)
	# print auc, tr_auc

	# cu.checkAndCreate(output_folder)
	for i in range(len(res_list)):
	# i = 4
	# with open('%s/gt_te_fold%d_t'%(output_folder,i),'wb') as f:
	# 	pickle.dump(gt_te,f)
	# with open('%s/pt_te_fold%d_t'%(output_folder,i),'wb') as f:
	# 	pickle.dump(pt_te,f)
	# with open('%s/pre_te_fold%d_t'%(output_folder,i),'wb') as f:
	# 	pickle.dump(res_list[i]['n_pre_te'],f)
		with open('%s/c_pre_te_fold%d'%(output_folder,i),'wb') as f:
			pickle.dump(res_list[i]['c_pre_te'],f)

	# print res['auc_te']

# for i in range(5):
# 	error_analysis(
# 		'../data/seed2222/raw/interp/mean',
# 		'../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_s005',
# 		i)
# error_analysis(
# 	'../data/seed2222/raw/interp/mean',
# 	'../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_outcomes')

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

plot_auc()

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

	# print temp_dic
	# print base_dic

tcmax = 6; 
tu = 60*2; # 2 hrs intervals, 12 hrs
toffset = 720; # toffset changes will affect has_last_12h
def has_last_12h(tidx):
    if len(tidx) > 1 and tidx[-1] >= toffset:
        return True
    else:
        return False

def repeat_last(tidx_new, tidx, vals):
    vals_new = zeros(tidx_new.shape)
    j = 0
    for i in range(len(tidx_new)):
        while j < len(tidx) and tidx[j] <= tidx_new[i]:
            vals_new[i] = vals[j]; j += 1
    return vals_new;

def interpolating(ptarr, sid, cz=False):
	tgraph = {}; tnode = {}; ntext = {}; etext = {}; hlab = {}
	(rows, cols) = ptarr.shape
	tidx = ptarr[:,0]; # list of timeindex

	# if not self.has_1st_day(tidx): #  , has_last_day
	#	 return (tgraph, tnode)
	if not has_last_12h(tidx):
		return None
	thridx = tidx[-1] - 720 + tu * (arange(tcmax) + 1)

	# thridx = self.toffset + self.tu * (arange(self.tcmax) + 1)
	# the line below offset to the last day
	# thridx = tidx[len(tidx)-1] - thridx[len(thridx)-1] + thridx

	thridx = thridx.reshape(tcmax, 1); iptarr = thridx
	# thridx = reshape([t1, t2, ..., tn], (n,1))

	# iptarr (patient array of certain times)
	# [time, 'Creatinine', 'BUN', ...] (features)
	# [t1, xxx, xxx, ...]
	# [t2, xxx, xxx, ...]
	# ...
	# [tn, xxx, xxx, ...]

	for ci in range(1,cols-2):
		y = ptarr[:,ci]				 # values of feature ci
		yi = interp(thridx, tidx, y)	# get interp for certain times in
										# mapping from time to feature value
		iptarr = hstack((iptarr, yi))
	# med_l, which doesn't make sense for linear interpolation
	y = ptarr[:,cols-2]
	yi = repeat_last(thridx, tidx, y)
	iptarr = hstack((iptarr, yi))
	# loc, which doesn't make sense for linear interpolation
	y = ptarr[:,cols-1]
	yi = repeat_last(thridx, tidx, y)
	iptarr = hstack((iptarr, yi))

	return iptarr

def scan_csv_interpolation(fnin, fout, cz):
	fin = open(fnin, 'r')
	freader = csv.reader(fin, delimiter=',', quotechar="\"")
	lcnt = 0; current_sid = ""; ptarr = None; gcnt = 0; current_mort = 0;
	# ptarr (patient array)
	# [time, 'Creatinine', 'BUN', ...]
	# [[0, xxx, xxx, ...],
	#  [1, xxx, xxx, ...],
	#  ...
	#  [last measure, xxx, xxx, ...]]
	hv = {}
	colname = ['timeindex', 'Creatinine_r', 'BUN_r', 'BUNtoCr_r', 'urineByHrByWeight_r', 'eGFR_r', 'AST_r', 'ALT_r', 'TBili_r', 'DBili_r', 'Albumin_r', 'tProtein_r', 'ASTtoALT_r', 'HCT_r', 'Hgb_r', 'INR_r', 'Platelets_r', 'PT_r', 'PTT_r', 'RBC_r', 'WBC_r', 'RESP_r', 'mSaO2_r', 'PaO2toFiO2_r', 'MinuteVent_r', 'DeliveredTidalVolume_r', 'FiO2Set_r', 'PEEPSet_r', 'PIP_r', 'RSBI_r', 'RSBIRate_r', 'RAW_r', 'PlateauPres_r', 'mSBP_r', 'mDBP_r', 'mMAP_r', 'CV_HR_r', 'mCrdIndx_r', 'mCVP_r', 'Art_BE_r', 'Art_CO2_r', 'Art_PaCO2_r', 'Art_PaO2_r', 'Art_pH_r', 'Na_r', 'K_r', 'Cl_r', 'Glucose_r', 'Ca_r', 'Mg_r', 'IonCa_r', 'Lactate_r', 'GCS_r', 'temp_r',
	'Antiarrhythmic_m', 'Anticoagulant_m', 'Antiplatelet_m', 'Benzodiazepine_m', 'beta_Blocking_m', 'Calcium_channel_blocking_m', 'Diuretic_m', 'Hemostatic_m', 'Inotropic_m', 'Insulin_m', 'Nondepolarizing_m', 'sedatives_m', 'Somatostatin_preparation_m', 'Sympathomimetic_m', 'Thrombolytic_m', 'Vasodilating_m',
	'AIDS_p', 'HemMalig_p', 'MetCarcinoma_p', 'medtype.label', 'location.label']
	df = pd.DataFrame(columns=colname)
	for row in freader:
		lcnt += 1
		if lcnt == 1:
			if cz:
				vns = ['readmit', 'sid', 'timeindex',
				'Creatinine_n', 'BUN_n', 'BUNtoCr_n', 'urineByHrByWeight_n', 'eGFR_n', 'AST_n', 'ALT_n', 'TBili_n',
				'DBili_n', 'Albumin_n', 'tProtein_n', 'ASTtoALT_n', 'HCT_n', 'Hgb_n', 'INR_n', 'Platelets_n', 'PT_n',
				'PTT_n', 'RBC_n', 'WBC_n', 'RESP_n', 'mSaO2_n', 'PaO2toFiO2_n', 'MinuteVent_n', 'DeliveredTidalVolume_n',
				'FiO2Set_n', 'PEEPSet_n', 'PIP_n', 'PlateauPres_n', 'RAW_n', 'RSBI_n', 'RSBIRate_n', 'mSBP_n', 'mDBP_n',
				'mMAP_n', 'CV_HR_n', 'mCrdIndx_n', 'mCVP_n', 'Art_BE_n', 'Art_CO2_n', 'Art_PaCO2_n', 'Art_PaO2_n',
				'Art_pH_n', 'Na_n', 'K_n', 'Cl_n', 'Glucose_n', 'Ca_n', 'Mg_n', 'IonCa_n', 'Lactate_n', 'GCS_n',
				'temp_n', 'Age_n', 
				'Antiarrhythmic_m', 'Anticoagulant_m', 'Antiplatelet_m', 'Benzodiazepine_m', 'beta_Blocking_m', 
				'Calcium_channel_blocking_m', 'Diuretic_m', 'Hemostatic_m', 'Inotropic_m', 'Insulin_m', 
				'Nondepolarizing_m', 'sedatives_m', 'Somatostatin_preparation_m', 'Sympathomimetic_m', 
				'Thrombolytic_m', 'Vasodilating_m', 'AIDS_p', 'HemMalig_p', 'MetCarcinoma_p', 'medtype.label', 
				'location.label']
			else:
				vns = ['readmit', 'sid', 'timeindex',
				'Creatinine_r', 'BUN_r', 'BUNtoCr_r', 'urineByHrByWeight_r', 'eGFR_r', 'AST_r', 'ALT_r', 'TBili_r', 'DBili_r',
				'Albumin_r', 'tProtein_r', 'ASTtoALT_r', 'HCT_r', 'Hgb_r', 'INR_r',
				'Platelets_r', 'PT_r', 'PTT_r', 'RBC_r', 'WBC_r', 'RESP_r', 'mSaO2_r', 'PaO2toFiO2_r', 'MinuteVent_r',
				'DeliveredTidalVolume_r', 'FiO2Set_r', 'PEEPSet_r', 'PIP_r', 'PlateauPres_r', 'RAW_r',
				'RSBI_r', 'RSBIRate_r', 'mSBP_r', 'mDBP_r', 'mMAP_r', 'CV_HR_r', 'mCrdIndx_r', 'mCVP_r', 'Art_BE_r',
				'Art_CO2_r', 'Art_PaCO2_r', 'Art_PaO2_r', 'Art_pH_r', 'Na_r', 'K_r', 'Cl_r', 'Glucose_r', 'Ca_r', 'Mg_r',
				'IonCa_r', 'Lactate_r', 'GCS_r', 'temp_r', 'Age_r',
				'Antiarrhythmic_m', 'Anticoagulant_m', 'Antiplatelet_m', 'Benzodiazepine_m', 'beta_Blocking_m', 
				'Calcium_channel_blocking_m', 'Diuretic_m', 'Hemostatic_m', 'Inotropic_m', 'Insulin_m', 
				'Nondepolarizing_m', 'sedatives_m', 'Somatostatin_preparation_m', 'Sympathomimetic_m', 
				'Thrombolytic_m', 'Vasodilating_m', 'AIDS_p', 'HemMalig_p', 'MetCarcinoma_p', 'medtype.label', 
				'location.label']
			_ = row
		else:
			for i in range(len(row)):
				if row[i] != "":
					row[i] = float(row[i])
				else:
					row[i] = NaN
				hv[vns[i]] = row[i]

			time = int(hv['timeindex']); sid = int(hv['sid'])

			if current_sid != "" and current_sid != sid:
				iptarr = interpolating(ptarr, current_sid, cz)
				df_iptarr = pd.DataFrame(iptarr,columns=colname)
				df_iptarr['sid'] = current_sid
				df = df.append(df_iptarr)
				current_sid = sid; ptarr = None
			if cz:
				arow = array([time,
					hv['Creatinine_n'], hv['BUN_n'], hv['BUNtoCr_n'], hv['urineByHrByWeight_n'], hv['eGFR_n'], hv['AST_n'], hv['ALT_n'], hv['TBili_n'], hv['DBili_n'], hv['Albumin_n'], hv['tProtein_n'], hv['ASTtoALT_n'], hv['HCT_n'], hv['Hgb_n'], hv['INR_n'], hv['Platelets_n'], hv['PT_n'], hv['PTT_n'], hv['RBC_n'], hv['WBC_n'], hv['RESP_n'], hv['mSaO2_n'], hv['PaO2toFiO2_n'], hv['MinuteVent_n'], hv['DeliveredTidalVolume_n'], hv['FiO2Set_n'], hv['PEEPSet_n'], hv['PIP_n'], hv['RSBI_n'], hv['RSBIRate_n'], hv['RAW_n'], hv['PlateauPres_n'], hv['mSBP_n'], hv['mDBP_n'], hv['mMAP_n'], hv['CV_HR_n'], hv['mCrdIndx_n'], hv['mCVP_n'], hv['Art_BE_n'], hv['Art_CO2_n'], hv['Art_PaCO2_n'], hv['Art_PaO2_n'], hv['Art_pH_n'], hv['Na_n'], hv['K_n'], hv['Cl_n'], hv['Glucose_n'], hv['Ca_n'], hv['Mg_n'], hv['IonCa_n'], hv['Lactate_n'], hv['GCS_n'], hv['temp_n'],
					# meds
					hv['Antiarrhythmic_m'], hv['Anticoagulant_m'], hv['Antiplatelet_m'], hv['Benzodiazepine_m'], hv['beta_Blocking_m'], hv['Calcium_channel_blocking_m'], hv['Diuretic_m'], hv['Hemostatic_m'], hv['Inotropic_m'], hv['Insulin_m'], hv['Nondepolarizing_m'], hv['sedatives_m'], hv['Somatostatin_preparation_m'], hv['Sympathomimetic_m'], hv['Thrombolytic_m'], hv['Vasodilating_m'],
					# malig
					hv['AIDS_p'], hv['HemMalig_p'], hv['MetCarcinoma_p'],
					# 
					hv['medtype.label'], hv['location.label']])
			else:
				arow = array([time,
					hv['Creatinine_r'], hv['BUN_r'], hv['BUNtoCr_r'], hv['urineByHrByWeight_r'], hv['eGFR_r'], hv['AST_r'], hv['ALT_r'], hv['TBili_r'], hv['DBili_r'], hv['Albumin_r'], hv['tProtein_r'], hv['ASTtoALT_r'], hv['HCT_r'], hv['Hgb_r'], hv['INR_r'], hv['Platelets_r'], hv['PT_r'], hv['PTT_r'], hv['RBC_r'], hv['WBC_r'], hv['RESP_r'], hv['mSaO2_r'], hv['PaO2toFiO2_r'], hv['MinuteVent_r'], hv['DeliveredTidalVolume_r'], hv['FiO2Set_r'], hv['PEEPSet_r'], hv['PIP_r'], hv['RSBI_r'], hv['RSBIRate_r'], hv['RAW_r'], hv['PlateauPres_r'], hv['mSBP_r'], hv['mDBP_r'], hv['mMAP_r'], hv['CV_HR_r'], hv['mCrdIndx_r'], hv['mCVP_r'], hv['Art_BE_r'], hv['Art_CO2_r'], hv['Art_PaCO2_r'], hv['Art_PaO2_r'], hv['Art_pH_r'], hv['Na_r'], hv['K_r'], hv['Cl_r'], hv['Glucose_r'], hv['Ca_r'], hv['Mg_r'], hv['IonCa_r'], hv['Lactate_r'], hv['GCS_r'], hv['temp_r'], 
					# meds
					hv['Antiarrhythmic_m'], hv['Anticoagulant_m'], hv['Antiplatelet_m'], hv['Benzodiazepine_m'], hv['beta_Blocking_m'], hv['Calcium_channel_blocking_m'], hv['Diuretic_m'], hv['Hemostatic_m'], hv['Inotropic_m'], hv['Insulin_m'], hv['Nondepolarizing_m'], hv['sedatives_m'], hv['Somatostatin_preparation_m'], hv['Sympathomimetic_m'], hv['Thrombolytic_m'], hv['Vasodilating_m'],
					# malig
					hv['AIDS_p'], hv['HemMalig_p'], hv['MetCarcinoma_p'],
					#
					# hv['Creatinine_n'], hv['BUN_n'], hv['BUNtoCr_n'], hv['urineByHrByWeight_n'], hv['eGFR_n'], hv['AST_n'], hv['ALT_n'], hv['TBili_n'], hv['DBili_n'], hv['Albumin_n'], hv['tProtein_n'], hv['ASTtoALT_n'], hv['HCT_n'], hv['Hgb_n'], hv['INR_n'], hv['Platelets_n'], hv['PT_n'], hv['PTT_n'], hv['RBC_n'], hv['WBC_n'], hv['RESP_n'], hv['mSaO2_n'], hv['PaO2toFiO2_n'], hv['MinuteVent_n'], hv['DeliveredTidalVolume_n'], hv['FiO2Set_n'], hv['PEEPSet_n'], hv['PIP_n'], hv['RSBI_n'], hv['RSBIRate_n'], hv['RAW_n'], hv['PlateauPres_n'], hv['mSBP_n'], hv['mDBP_n'], hv['mMAP_n'], hv['CV_HR_n'], hv['mCrdIndx_n'], hv['mCVP_n'], hv['Art_BE_n'], hv['Art_CO2_n'], hv['Art_PaCO2_n'], hv['Art_PaO2_n'], hv['Art_pH_n'], hv['Na_n'], hv['K_n'], hv['Cl_n'], hv['Glucose_n'], hv['Ca_n'], hv['Mg_n'], hv['IonCa_n'], hv['Lactate_n'], hv['GCS_n'], hv['temp_n'],
					hv['medtype.label'], hv['location.label']])
			arow = arow.reshape(1, arow.size)
			if ptarr is None:
				ptarr = arow
			else:
				ptarr = vstack((ptarr,arow))

			current_sid = sid; current_mort = int(hv['readmit'])
	iptarr = interpolating(ptarr, current_sid, cz)
	df_iptarr = pd.DataFrame(iptarr,columns=colname)
	df_iptarr['sid'] = current_sid
	df = df.append(df_iptarr)

	df.to_csv(fout,index=False)
	fin.close()

def plot_trends(fn_ori, fn_imp, fout, patient_idlist=None):
	scan_csv_interpolation(
		fn_imp,
		'../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_s005/plotting/test_fold0_6slots.csv',
		False)
	data_ori = pd.read_csv('../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_s005/plotting/test_fold0_6slots.csv')
	if patient_idlist == None:
		patient_idlist = data_ori['sid'].unique().tolist()
		print patient_idlist
		print len(patient_idlist)
	for ptid in patient_idlist:
	# for ptid in [138]:
		gp_ori = data_ori.groupby('sid')

		pt_ori = gp_ori.get_group(ptid)

		X = pt_ori['timeindex'].tolist()
		# print X

		lm_dic = {}

		for col in pt_ori.columns:
		# for col in labs:
			Y = array([])
			plt.clf()
			lm = 0
			for i in pt_ori.index.values:
				v = pt_ori[col][i]
				if v <= 1 and v >= -1:
					v = 0
				elif v <= 2 and v > 1:
					v = 1
				elif v < -1 and v >= -2:
					v = -1
				elif v > 2:
					v = 2
				elif v < -2:
					v = -2
				lm = v
				Y = append(Y,v)

			plt.ylim([-3,3])
			plt.scatter(X,Y,c='r')
			plt.title(col)
			plt.plot(X,Y)
			
			if not os.path.exists('%s/%d/med'%(fout,ptid)):
				os.makedirs('%s/%d/med'%(fout,ptid))
			# med_cols = ['Antiarrhythmic_m', 'Anticoagulant_m', 'Antiplatelet_m', 'Benzodiazepine_m', 'beta_Blocking_m', 'Calcium_channel_blocking_m', 'Diuretic_m', 'Hemostatic_m', 'Inotropic_m', 'Insulin_m', 'Nondepolarizing_m', 'sedatives_m', 'Somatostatin_preparation_m', 'Sympathomimetic_m', 'Thrombolytic_m', 'Vasodilating_m', 'AIDS_p', 'HemMalig_p', 'MetCarcinoma_p', 'medtype.label', 'location.label']
			if col in med_cols:
				plt.savefig('%s/%d/med/%d_%s.png' % (fout, ptid, ptid, col))
			elif col in labs:
				plt.savefig('%s/%d/%d_%s.png' % (fout, ptid, ptid, col))
			else:
				print col

			if col in labs:
				if lm in lm_dic:
					lm_dic[lm] += 1
				else:
					lm_dic[lm] = 1

		with open('%s/%d/lm_dic'%(fout,ptid),'wb') as f:
			pickle.dump(lm_dic,f)

		print ptid
		print lm_dic
			
# labs=['Creatinine_r', 'BUN_r', 'BUNtoCr_r', 'urineByHrByWeight_r', 'eGFR_r', 'AST_r', 'ALT_r', 'TBili_r', 'DBili_r', 'Albumin_r', 'tProtein_r', 'ASTtoALT_r', 'HCT_r', 'Hgb_r', 'INR_r', 'Platelets_r', 'PT_r', 'PTT_r', 'RBC_r', 'WBC_r', 'RESP_r', 'mSaO2_r', 'PaO2toFiO2_r', 'MinuteVent_r', 'DeliveredTidalVolume_r', 'FiO2Set_r', 'PEEPSet_r', 'PIP_r', 'RSBI_r', 'RSBIRate_r', 'RAW_r', 'PlateauPres_r', 'mSBP_r', 'mDBP_r', 'mMAP_r', 'CV_HR_r', 'mCrdIndx_r', 'mCVP_r', 'Art_BE_r', 'Art_CO2_r', 'Art_PaCO2_r', 'Art_PaO2_r', 'Art_pH_r', 'Na_r', 'K_r', 'Cl_r', 'Glucose_r', 'Ca_r', 'Mg_r', 'IonCa_r', 'Lactate_r', 'GCS_r', 'temp_r']
# med_cols = ['Antiarrhythmic_m', 'Anticoagulant_m', 'Antiplatelet_m', 'Benzodiazepine_m', 'beta_Blocking_m', 'Calcium_channel_blocking_m', 'Diuretic_m', 'Hemostatic_m', 'Inotropic_m', 'Insulin_m', 'Nondepolarizing_m', 'sedatives_m', 'Somatostatin_preparation_m', 'Sympathomimetic_m', 'Thrombolytic_m', 'Vasodilating_m', 'AIDS_p', 'HemMalig_p', 'MetCarcinoma_p', 'medtype.label', 'location.label']
# cu.checkAndCreate('../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_s005/plotting')
# plot_trends('../data/seed2222/raw/test_fold0_z.csv', 
# '../data/seed2222/raw/mice/mp0.5_mc0.6/dataset/test_fold0_z.csv', 
# '../observer/error_analysis/seed2222_raw_mice_mp0.5_mc0.6_z_isg0_s006/plotting',
# [21,402,631,1840,4749,22553,5908,22092,24958,10451,25384,4271])
# plot_trends('../data/seed2222/raw/test_fold0_z.csv', 
# '../data/seed2222/raw/interp/mean/dataset/test_fold0_z.csv', 
# '../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_s005/plotting',
# [3602,17122,18094,2333,5581,18022,6212,18420])
# [17795,19334,22579,5278]
# interp_mean
# 17795: (0, 0, 1)
# 3602:  (0, 0, 1)
# 17122: (0, 0, 1)

# 19334: (0, 1, 0)
# 18094: (0, 1, 0)
# 2333:  (0, 1, 0)

# 22579: (1, 0, 1)
# 5581:  (1, 0, 1)
# 18022: (1, 0, 1)

# 5278:  (1, 1, 0)
# 6212:  (1, 1, 0)
# 18420: (1, 1, 0)
# [21,402,631,1840]
# sid_pred_diff_btw_baseline_temporal()
# error_analysis()


def check_last12h_imputed_value(fn, fnout):
	out = ''
	df = pd.read_csv(fn)
	gp = df.groupby('sid')
	for sid,g in gp:
		start_tidx = g['timeindex'].iloc[-1] - 720

		if start_tidx < 0:
			continue
		last12hData = g[g['timeindex']>=start_tidx]
		for col in last12hData.columns:
			if last12hData[col].isnull().sum() == last12hData.shape[0] and g[col].isnull().sum() != g.shape[0]:
				print sid, col
				out += '%d,%s\n'%(sid,col)

	fout = open(fnout,'w')
	fout.write(out)
	fout.close()

# check_last12h_imputed_value('../data/alldata_readmit.csv','../observer/check_last12h_imputed_value.txt')

# last12hData = pd.DataFrame(columns=df.columns)
# gp = df.groupby('sid')
# for sid,g in gp:
# 	# tidx = g['timeindex']
# 	tidx = array(g['timeindex'])
# 	start_tidx = tidx[-1] - 720
	
# 	if has_last_12h(tidx):
# 		last12hData = last12hData.append(g[g['timeindex']>=start_tidx])
# 	# if sid == 21:
# 	# 	break
# last12hData.to_csv('../data/alldata_readmit_last12h.csv',index=False)

def plot_variable(v, fout):
	te = pd.read_csv('../data/seed2222/raw/test_fold0.csv')
	tr = pd.read_csv('../data/seed2222/raw/train_fold0.csv')
	data = tr.append(te)
	
	maxY = np.nanmax(data[v].tolist())
	minY = np.nanmin(data[v].tolist())
	ylml = -1000
	ylmh = 1000
	
	gp = data.groupby('sid')
	for sid,g in gp:
		X = np.array([])
		Y = np.array([])
		c = np.array([])
		plt.clf()
		for i in g.index.values:
			if not np.isnan(g[v][i]):
				X = np.append(X,g['timeindex'][i])
				Y = np.append(Y,g[v][i])
				c = np.append(c,'g')
		
		fig = plt.figure()
		fig.suptitle(v, fontsize=14, fontweight='bold')
		ax = fig.add_subplot(111)
		
		ax.set_ylim(ylml,ylmh)
		ax.scatter(X,Y,c=c)
		ax.plot(X,Y)
		
		if not os.path.exists('%s'%(fout)):
			os.makedirs('%s/'%(fout))
		plt.savefig('%s/%s_%s.png' % (fout, sid, v))

# if __name__ == '__main__':
	# plot_variable('ALT', '../observer/error_analysis/ALT_plots_obsTimepoint_scale')
	# badptlist = ['8686','20064'] # <0.2 or >0.8
	# badptlist = ['843', '1516', '16791', '17827', '19098', '26475'] #<0.25 or >0.85
	# edgeptlist = ['8110', '16805', '18054', '20548', '21166', '22663', '25954']

	# l = [19148]
	# plot_imputes_values(l,'',istrain=True)

	# plot_imputes_values(badptlist,'../observer/error_analysis/t+b_plot/bad')
	# plot_imputes_values(edgeptlist,'../observer/error_analysis/t+b_plot/edge')

	# error_analysis('../data/seed2222/raw/interp/mean',
	# 	'../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_outcomes')
	# test_time('../data/seed2222/raw/interp/mean')

# correctptlist=[]
# for i in range(len(pt_te_fold4)):
# 	if gt_te_fold4[i] == c_pre_te_fold4[i]:
# 		correctptlist.append(pt_te_fold4[i])

# wrongptlist = []
# for i in range(len(pt_te_fold4)):
# 	if gt_te_fold4[i] != c_pre_te_fold4[i]:
# 		wrongptlist.append(pt_te_fold4[i])

# dic = {}
# for i in range(247):
# 	sid = str(int(te['sid'][i]))
# 	timeindex = te['timeindex'][i]
# 	if sid in pt_te_fold4:
# 		day = int(math.floor(timeindex/1440))
# 		if day not in dic:
# 			dic[day] = []
# 		dic[day].append(sid)

# cnt_dic = {}
# for k in dic:
# 	cnt_dic[k] = [0,0]

# for k in dic:
# 	for sid in dic[k]:
# 		if sid in correctptlist:
# 			cnt_dic[k][0] += 1
# 		elif sid in wrongptlist:
# 			cnt_dic[k][1] += 1

	# error_analysis(
	# 	'../data/seed2222/raw/interp/mean',
	# 	'../observer/error_analysis/seed2222_raw_interp_mean_z_isg0_outcomes')