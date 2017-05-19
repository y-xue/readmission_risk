import pandas as pd
import numpy as np
import subprocess
import re
import pickle
import os
import math

from scipy import interp
from sklearn import metrics
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import mimic_tensor as mt
import mimic_data as md
import mimic_classifying as mc
import nel_graph_gen_interpolation as nggi
import coding_util as cu

import multiprocessing

standardized_features_for_classify = ['readmit', 'sid', 'timeindex', 'Creatinine.standardized', 'BUN.standardized', 
'BUNtoCr.standardized', 'urineByHrByWeight.standardized', 'eGFR.standardized', 'AST.standardized', 
'ALT.standardized', 'TBili.standardized', 'DBili.standardized', 'Albumin.standardized', 
'tProtein.standardized', 'ASTtoALT.standardized', 'HCT.standardized', 'Hgb.standardized', 
'INR.standardized', 'Platelets.standardized', 'PT.standardized', 'PTT.standardized', 'RBC.standardized', 
'WBC.standardized', 'RESP.standardized', 'mSaO2.standardized', 'PaO2toFiO2.standardized', 
'MinuteVent.standardized', 'DeliveredTidalVolume.standardized', 'FiO2Set.standardized', 
'PEEPSet.standardized', 'PIP.standardized', 'PlateauPres.standardized', 'RAW.standardized', 
'RSBI.standardized', 'RSBIRate.standardized', 'mSBP.standardized', 'mDBP.standardized', 
'mMAP.standardized', 'CV_HR.standardized', 'mCrdIndx.standardized', 'mCVP.standardized', 
'Art_BE.standardized', 'Art_CO2.standardized', 'Art_PaCO2.standardized', 'Art_PaO2.standardized', 
'Art_pH.standardized', 'Na.standardized', 'K.standardized', 'Cl.standardized', 'Glucose.standardized', 
'Ca.standardized', 'Mg.standardized', 'IonCa.standardized', 'Lactate.standardized', 'GCS.standardized', 
'temp.standardized', 'Age.standardized', 'Antiarrhythmic_agent', 'Anticoagulant', 'Antiplatelet_agent', 
'Benzodiazepine', 'beta.Blocking_agent', 'Calcium_channel_blocking_agent', 'Diuretic', 'Hemostatic_agent', 
'Inotropic_agent', 'Insulin', 'Nondepolarizing_agent', 'sedatives', 'Somatostatin_preparation', 
'Sympathomimetic_agent', 'Thrombolytic_agent', 'Vasodilating_agent', 'AIDS', 'HemMalig', 'MetCarcinoma', 
'medtype.label', 'location.label']

standardized_features_for_impute = ['Creatinine.standardized', 'BUN.standardized', 
'BUNtoCr.standardized', 'urineByHrByWeight.standardized', 'eGFR.standardized', 'AST.standardized', 
'ALT.standardized', 'TBili.standardized', 'DBili.standardized', 'Albumin.standardized', 
'tProtein.standardized', 'ASTtoALT.standardized', 'HCT.standardized', 'Hgb.standardized', 
'INR.standardized', 'Platelets.standardized', 'PT.standardized', 'PTT.standardized', 'RBC.standardized', 
'WBC.standardized', 'RESP.standardized', 'mSaO2.standardized', 'PaO2toFiO2.standardized', 
'MinuteVent.standardized', 'DeliveredTidalVolume.standardized', 'FiO2Set.standardized', 
'PEEPSet.standardized', 'PIP.standardized', 'PlateauPres.standardized', 'RAW.standardized', 
'RSBI.standardized', 'RSBIRate.standardized', 'mSBP.standardized', 'mDBP.standardized', 
'mMAP.standardized', 'CV_HR.standardized', 'mCrdIndx.standardized', 'mCVP.standardized', 
'Art_BE.standardized', 'Art_CO2.standardized', 'Art_PaCO2.standardized', 'Art_PaO2.standardized', 
'Art_pH.standardized', 'Na.standardized', 'K.standardized', 'Cl.standardized', 'Glucose.standardized', 
'Ca.standardized', 'Mg.standardized', 'IonCa.standardized', 'Lactate.standardized', 'GCS.standardized', 
'temp.standardized', 'Age.standardized']

raw_features_for_classify = ['readmit', 'sid', 'timeindex', 'Creatinine', 'BUN', 'BUNtoCr', 
'urineByHrByWeight', 'eGFR', 'AST', 'ALT', 'TBili', 'DBili', 'Albumin', 'tProtein', 
'ASTtoALT', 'HCT', 'Hgb', 'INR', 'Platelets', 'PT', 'PTT', 'RBC', 'WBC', 'RESP', 
'mSaO2', 'PaO2toFiO2', 'MinuteVent', 'DeliveredTidalVolume', 'FiO2Set', 'PEEPSet', 
'PIP', 'PlateauPres', 'RAW', 'RSBI', 'RSBIRate', 'mSBP', 'mDBP', 'mMAP', 'CV_HR', 
'mCrdIndx', 'mCVP', 'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_PaO2', 'Art_pH', 'Na', 
'K', 'Cl', 'Glucose', 'Ca', 'Mg', 'IonCa', 'Lactate', 'GCS', 'temp', 'Age', 
'Antiarrhythmic_agent', 'Anticoagulant', 'Antiplatelet_agent', 'Benzodiazepine', 
'beta.Blocking_agent', 'Calcium_channel_blocking_agent', 'Diuretic', 'Hemostatic_agent', 
'Inotropic_agent', 'Insulin', 'Nondepolarizing_agent', 'sedatives', 
'Somatostatin_preparation', 'Sympathomimetic_agent', 'Thrombolytic_agent', 
'Vasodilating_agent', 'AIDS', 'HemMalig', 'MetCarcinoma', 'medtype.label', 'location.label']

raw_features_for_impute = ['Creatinine', 'BUN', 'BUNtoCr', 
'urineByHrByWeight', 'eGFR', 'AST', 'ALT', 'TBili', 'DBili', 'Albumin', 'tProtein', 
'ASTtoALT', 'HCT', 'Hgb', 'INR', 'Platelets', 'PT', 'PTT', 'RBC', 'WBC', 'RESP', 
'mSaO2', 'PaO2toFiO2', 'MinuteVent', 'DeliveredTidalVolume', 'FiO2Set', 'PEEPSet', 
'PIP', 'PlateauPres', 'RAW', 'RSBI', 'RSBIRate', 'mSBP', 'mDBP', 'mMAP', 'CV_HR', 
'mCrdIndx', 'mCVP', 'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_PaO2', 'Art_pH', 'Na', 
'K', 'Cl', 'Glucose', 'Ca', 'Mg', 'IonCa', 'Lactate', 'GCS', 'temp', 'Age']

# exlude_features = ['readmit', 'sid', 'timeindex',
# 'Antiarrhythmic_agent', 'Anticoagulant', 'Antiplatelet_agent', 
# 'Benzodiazepine', 'beta.Blocking_agent', 'Calcium_channel_blocking_agent', 'Diuretic', 'Hemostatic_agent', 
# 'Inotropic_agent', 'Insulin', 'Nondepolarizing_agent', 'sedatives', 'Somatostatin_preparation', 
# 'Sympathomimetic_agent', 'Thrombolytic_agent', 'Vasodilating_agent', 'AIDS', 'HemMalig', 'MetCarcinoma', 
# 'medtype.label', 'location.label']

# change cdn in mimic_data.py as well!!!

nel_graph_length = 13

freq_list = ['001','002','003','004','005','006','008','009', '01','011']
freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}

# without med_feature
freq_list_no_med = ['001','002','004','006','007','009','010', '012','014','015']

def clean_data(self, fn, fout):
	'''
	remove values in standardized features 
	whose corresponding value in raw features is missing.
	
	e.g. pt1['BUN'] = NaN, pt1['BUN_n'] = 1
	we should make pt1['BUN_n'] NaN
	'''
	data = pd.read_csv(fn)
	rows = data.shape[0]
	for i in range(rows):
		for col in data.columns:
			if 'standardized' in col:
				continue
			col_n = col + '.standardized'
			if col_n not in data.columns:
				continue
			if pd.isnull(data[col][i]) and not pd.isnull(data[col_n][i]):
				data = data.set_value(i,col_n,np.nan)
	col_names = data.columns.tolist()
	col_names[0] = 'readmit'
	data.columns = col_names
	data.to_csv(fout,index=False)

def split_by_feature_type(cdn, fn_prefix, raw_colname, z_colname):
	'''
	split data into two sets, one contains raw features + medical features, 
	another contains standardized features + medical features.
	'''
	for i in range(5):
		training = pd.read_csv('%s_train_fold%d.csv'%(fn_prefix,i))
		testing = pd.read_csv('%s_test_fold%d.csv'%(fn_prefix,i))
		raw_train = training[raw_colname]
		raw_test = testing[raw_colname]
		z_train = training[z_colname]
		z_test = testing[z_colname]
		cu.checkAndCreate('%s/raw/'%cdn)
		cu.checkAndCreate('%s/z/'%cdn)
		raw_train.to_csv('%s/raw/train_fold%d.csv'%(cdn,i),index=False)
		raw_test.to_csv('%s/raw/test_fold%d.csv'%(cdn,i),index=False)
		z_train.to_csv('%s/z/train_fold%d.csv'%(cdn,i),index=False)
		z_test.to_csv('%s/z/test_fold%d.csv'%(cdn,i),index=False)

def split_test_by_patient(tcdn):
	'''
	This is a help function for MICE imputation.
	Given folder, extract data for each patient to be used in MICE
	'''
	for i in range(5):
		test = pd.read_csv('%s/test_fold%d.csv'%(tcdn,i))
		gp = test.groupby('sid')
		cu.checkAndCreate('%s/mice/imputing/test_fold%d'%(tcdn,i))
		fn = open('%s/mice/imputing/test_fold%d/sid_list.txt'%(tcdn,i),'w')
		for sid, group in gp:
			group.to_csv('%s/mice/imputing/test_fold%d/%d.csv'%(tcdn,i,sid),index=False)
			fn.write('%d\n'%sid)
		fn.close()

def impute_by_mean(ftr, fte, fimptr, fimpte, meth='mean'):
	'''
	Given original train/test sets (ftr, fte), imputes with mean value.
	Then write to fimptr and fimpte.
	'''
	print 'imputing by mean values...'

	if meth == 'mean':
		training = pd.read_csv(ftr)
		testing = pd.read_csv(fte)
		mtr = training.fillna(training.mean())
		mte = testing.fillna(training.mean())
		mtr.to_csv(fimptr, index=False)
		mte.to_csv(fimpte, index=False)

class Experiment():

	def __init__(self, cdn, seed):
		'''
		cdn = '../data/seed%s/%s/mice/mp%s_mc%s'%(seed,ft,minp,minc)
		'''
		self.seed = seed
		self.cdn = cdn
		self.ftrain = '%s/train_fold%d.csv'
		self.ftest = '%s/test_fold%d.csv'

	def interpolation(trcsv, tecsv, ftrnel, ftrnode, fnel, fnode):
		'''
		Do interpolation on training set (trcsv) and train+test set (trcsv+tecsv).
		Outputs: ftrnel, ftrnode, fnel, fnode.
		'''
		print 'interpolating...'
		trtecsv = '%s/trtecsv.csv'%self.cdn

		x = nggi.NelGraphGenInterpolation()
		x.scan_csv_interpolation(trcsv, ftrnel, ftrnode)

		tr = pd.read_csv(trcsv)
		te = pd.read_csv(tecsv)
		tr = tr.append(te)
		tr.to_csv(trtecsv, index=False)
		x.scan_csv_interpolation(trtecsv, fnel, fnode)

	def subgraph_mining(tr_nel, tr_te_nel, freq_t, foldi):
		'''
		Inputs: .nel file for training set (tr_nel) and training+testing set (tr_te_nel),
				MoSS param (freq_t), fold number.
		Outpus: .out and .ids files from MoSS

		Find frequent subgraphs (A) in training set first, then find frequent subgraphs
		from training+tesing set but only select those exists in A.
		'''
		print 'subgraph mining...'

		fntr = '%s/mimic_m1_train_fold%d'%(self.cdn,foldi)
		fntrte = '%s/mimic_m1_tr_te_fold%d'%(self.cdn,foldi)
		# fntr = '../data/testsubgraph/mimic_m1_train_fold%d'%(foldi)
		# fntrte = '../data/testsubgraph/mimic_m1_tr_te_fold%d'%(foldi)
		
		tr_freq_t = freq_to_trainFreq_map[freq_t]

		moss(tr_nel, fntr, tr_freq_t)
		hout_tr = read_fnout('%s.out'%(fntr))
		hids_tr = read_fnids('%s.ids'%(fntr))

		moss(tr_te_nel, fntrte, freq_t)
		hout_tr_te = read_fnout('%s.out'%(fntrte))
		hids_tr_te = read_fnids('%s.ids'%(fntrte))

		hout = {}
		hids = {}

		for k in hout_tr_te:
			if k in hout_tr:
				hout[hout_tr[k]['gid']] = k + hout_tr[k]['gstr'] + hout_tr_te[k]['sstr']
				hids[hout_tr[k]['gid']] = hids_tr_te[hout_tr_te[k]['gid']]

		write_subgraphs(hout, hids, '%s/mimic_m1_fold%d.out'%(self.cdn,foldi), '%s/mimic_m1_fold%d.ids'%(self.cdn,foldi))
		# write_subgraphs(hout, hids, '../data/testsubgraph/mimic_m1_fold%d.out'%(foldi), '../data/testsubgraph/mimic_m1_fold%d.ids'%(foldi))

	def gen_pt_sg_files(isg, freq_t, foldi):
		'''
		Generate patient_subgraph matrix.
		'''
		print 'gen_pt_sg_files:'
		print isg, freq_t, foldi

		fn_pt_sg_mat = "%s/isg%d/pt_sg_w/mimic_pt_sg.spmat_%s_fold%d"%(self.cdn,isg,freq_t,foldi)
		fn_pt_wd_mat = "%s/isg%d/pt_sg_w/mimic_pt_w.spmat_%s_fold%d"%(self.cdn,isg,freq_t,foldi)
		fn_sgs = "%s/isg%d/pt_sg_w/mimic.sgstr_%s_fold%d"%(self.cdn,isg,freq_t,foldi)
		fn_sparse_tensor = "%s/isg%d/pt_sg_w/mimic.tensor_fold%d"%(self.cdn,isg,foldi)
		fn_pt_gt = "%s/isg%d/pt_sg_w/mimic.ptmc_%s_fold%d" % (self.cdn,isg,freq_t,foldi)
		fn_pt_lab = "%s/isg%d/pt_sg_w/mimic.ptid_%s_fold%d"%(self.cdn,isg,freq_t,foldi)

		mt.pt_sg_w_tensor_gen("%s/mimic_fold%d.nel"%(self.cdn,foldi), 
			"%s/mimic_fold%d.node"%(self.cdn,foldi), 
			"%s/mimic_m1_fold%d.out"%(self.cdn,foldi), 
			"%s/mimic_m1_fold%d.ids"%(self.cdn,foldi), 
			fn_sparse_tensor, 
			fn_pt_sg_mat, 
			fn_pt_wd_mat, 
			fn_pt_lab, 
			"%s/isg%d/pt_sg_w/mimic.sgtid_%s_fold%d"%(self.cdn,isg,freq_t,foldi), 
			fn_sgs, 
			"%s/isg%d/pt_sg_w/mimic.ntid_%s_fold%d"%(self.cdn,isg,freq_t,foldi), isg=isg)

		ptsg = md.read_pt_sg_mat(fn_pt_sg_mat)
		ptwd = md.read_pt_wd_mat(fn_pt_wd_mat)
		sgs = md.read_sgs(fn_sgs)
		spt = md.read_sparse_tensor(fn_sparse_tensor)
		(hiso, hsgstr, hsgc, hsgsize) = md.sg_subiso(sgs)
		(ptsg, sgs, sptsel) = md.filter_sg(ptsg, hiso, sgs, spt=spt)
		gt = md.read_pt_gt(fn_pt_gt)
		pt = md.read_pt_lab(fn_pt_lab)
		write_pt_sg_before_filter_pt(ptsg, ptwd, sgs, pt, gt, isg, freq_t, foldi)

	# def classify_old(isg, s, ntestth, i, minp, minc):
	# 	'''
	# 	Direct classify using pt_sg.
	# 	'''
	# 	print 'classify:'
	# 	print 'minp = %s'%minp, 'minc = %s'%minc
	# 	print 'isg = %d'%isg, 's = %s'%s, 'ntestth = %d'%ntestth, 'fold = %d'%i
		
	# 	ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(self.cdn,isg,s,i),'rb'))
	# 	ptwd = pickle.load(open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(self.cdn,isg,s,i),'rb'))
	# 	sgs = pickle.load(open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(self.cdn,isg,s,i),'rb'))
	# 	pt = pickle.load(open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(self.cdn,isg,s,i),'rb'))
	# 	gt = pickle.load(open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(self.cdn,isg,s,i),'rb'))
	# 	(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)
	# 	(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
	# 		"../data/train/last12h_pmm_mp%s_mc%s_fold%d.csv"%(minp,minc,i), 
	# 		"../data/test/fold%d_mp%s_mc%s.csv"%(i,minp,minc))

	# 	for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
	# 		for pl in ['l1','l2']:
	# 			for cw in ['balanced', None]:
	# 				clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
	# 				res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	# 				with open('%s/isg%d/res/res_%s_nt%d_c%s_%s_%s_fold%d'%(self.cdn,isg,s,ntestth,str(c),pl,cw,i),'wb') as f:
	# 					pickle.dump(res,f)

	def dirClassify(ptsg, ptwd, sgs, pt, gt, ntestth, foldi, c, pl, cw):
		'''
		Direct classify using pt_sg.
		'''
		print 'classify:'
		
		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)
		(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
			self.ftrain%(self.cdn,foldi),
			self.ftest%(self.cdn,foldi))

		clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
		res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
		
		return res

	def nmfClassify(ptsg, ptwd, sgs, pt, gt, fnpik, ntestth, foldi, nc, c, pl, cw):
		'''
		Classify using NMF.
		'''
		print 'nmfclassify:'
		
		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)

		(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
			self.ftrain%(self.cdn,foldi),
			self.ftest%(self.cdn,foldi))
		
		clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
		
		(m, clf, res) = mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
			fnfmt=fnpik, nc=nc, clf=clf)

		return res

	def mean_auc(res):
		'''
		Calculate mean AUC of all folds.
		'''
		mean_tpr, mean_tpr_tr = 0.0, 0.0
		mean_fpr, mean_fpr_tr = np.linspace(0, 1), np.linspace(0, 1)
		for i in range(5):
			mean_tpr += interp(mean_fpr, res['fpr_te'], res['tpr_te'])
			mean_tpr_tr += interp(mean_fpr_tr, res['fpr_tr'], res['tpr_tr'])
			mean_tpr[0] = 0.0
			mean_tpr_tr[0] = 0.0
		mean_tpr /= 5
		mean_tpr_tr /= 5
		mean_tpr[-1] = 1.0
		mean_tpr_tr[-1] = 1.0
		mean_auc = metrics.auc(mean_fpr, mean_tpr)
		mean_auc_tr = metrics.auc(mean_fpr_tr, mean_tpr_tr)

		return (mean_auc, mean_auc_tr)

	def read_prediction_matrics(isg,freq_t):
		prediction_matrics = {}
		prediction_matrics['ptsg'] = []
		prediction_matrics['ptwd'] = []
		prediction_matrics['sgs'] = []
		prediction_matrics['pt'] = []
		prediction_matrics['gt'] = []
		for foldi in range(5):
			ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(self.cdn,isg,freq_t,foldi),'rb'))
			ptwd = pickle.load(open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(self.cdn,isg,freq_t,foldi),'rb'))
			sgs = pickle.load(open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(self.cdn,isg,freq_t,foldi),'rb'))
			pt = pickle.load(open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(self.cdn,isg,freq_t,foldi),'rb'))
			gt = pickle.load(open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(self.cdn,isg,freq_t,foldi),'rb'))

			prediction_matrics['ptsg'].append(ptsg)
			prediction_matrics['ptwd'].append(ptwd)
			prediction_matrics['sgs'].append(sgs)
			prediction_matrics['pt'].append(pt)
			prediction_matrics['gt'].append(gt)

		return prediction_matrics

	def tuneSGParamForClassification(nmf=False):
		output = ''
		for isg in [0,3]:
			output += 'isg %d: '%isg
			if nmf:
				cu.checkAndCreate('%s/isg%d/nmf_piks'%(self.cdn,isg))
			
			for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
				
				prediction_matrics = read_prediction_matrics(isg,freq_t)

				if nmf:
					bauc = 0.
					tbtauc = 0.
					bparams = ''
					for nc in [10,20,30,40,50,60,70,80,90,100,110,120]:
						(tbauc,tbparams,tbtauc) = tuneCLFParamForClassification(l, prediction_matrics, nmf=True, nc=nc)
						if tbauc > bauc:
							bauc = tbauc
							bparams = tbparams
							btauc = tbtauc
				else:
					(bauc,bparams,btauc) = tuneCLFParamForClassification(l, prediction_matrics, nmf=False)
			
			output += '%f (%s)\n'%(bauc,bparams)

		if nmf:
			fn = open('%s/nmfResult.txt'%(cdn),'w')
		else:
			fn = open('%s/dirResult.txt'%(cdn),'w')
		fn.write(output)
		fn.close()

	def tuneCLFParamForClassification(freq_t, prediction_matrics, nmf=False, nc=10):
		bauc = 0.
		htauc = 0.
		bparams = ''
		fparams = 's%s,nc%s,c%s,%s,%s'

		for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
			for pl in ['l1','l2']:
				for cw in ['balanced', None]:
					res_list = []
					for foldi in range(5):
						
						if nmf:
							res = nmfClassify(prediction_matrics['ptsg'][foldi],
								prediction_matrics['ptwd'][foldi],
								prediction_matrics['sgs'][foldi],
								prediction_matrics['pt'][foldi],
								prediction_matrics['gt'][foldi],
								'%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(self.cdn,isg,freq_t,foldi,nc),
								ntestth, foldi, nc, c, pl, cw)
						else:
							res = dirClassify(prediction_matrics['ptsg'][foldi],
								prediction_matrics['ptwd'][foldi],
								prediction_matrics['sgs'][foldi],
								prediction_matrics['pt'][foldi],
								prediction_matrics['gt'][foldi],
								ntestth, foldi, c, pl, cw)
						res_list.append(res)
					(auc, tr_auc) = mean_auc(res_list)

					if auc > bauc:
						bauc = auc
						bparams = fparams%(freq_t,nc,c,pl,cw)
					htauc = max(htauc,tr_auc)
		return (bauc, bparams, htauc)
	# def nmfclassify_old(isg, s, ntestth, i, nc=10, minp='0', minc='0', cdn=None):
	# 	'''
	# 	Classify using NMF.
	# 	'''
	# 	print 'nmfclassify:'
	# 	print '(isg=%d,s=%s,ntestth=%d,fold=%d,nc=%d):'%(isg,s,ntestth,i,nc)
		
	# 	ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(cdn,isg,s,i),'rb'))
	# 	ptwd = pickle.load(open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(cdn,isg,s,i),'rb'))
	# 	sgs = pickle.load(open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(cdn,isg,s,i),'rb'))
	# 	pt = pickle.load(open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(cdn,isg,s,i),'rb'))
	# 	gt = pickle.load(open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(cdn,isg,s,i),'rb'))
		
	# 	(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)

	# 	(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
	# 		'%s/alldata_readmit_last12h_mean_train_fold%d.csv'%(cdn,i),
	# 		'%s/alldata_readmit_last12h_mean_test_fold%d.csv'%(cdn,i))

	# 	for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
	# 		for pl in ['l1','l2']:
	# 			for cw in ['balanced', None]:
	# 				clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
					
	# 				(m, clf, res) = mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
	# 					fnfmt='%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(cdn,isg,s,i,nc), nc=nc, clf=clf)

	# 				if not os.path.exists('%s/isg%d/nmfres/%d'%(cdn,isg,nc)):
	# 					os.makedirs('%s/isg%d/nmfres/%d'%(cdn,isg,nc))
	# 				with open('%s/isg%d/nmfres/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn,isg,nc,s,ntestth,c,pl,cw,i),'wb') as f:
	# 					pickle.dump(res,f)

	# def mean_auc_old(isg, s, ntestth, c, pl, cw, nc=10, nmf=False,cdn=None):
	# 	'''
	# 	Calculate mean AUC of all folds.
	# 	'''
	# 	print '(isg=%d,s=%s,ntestth=%d):'%(isg,s,ntestth)
	# 	output = ''
	# 	best_auc = 0
	# 	best_output = ''

	# 	highest_tr_auc = 0
	# 	highest_tr_output = ''
	# 	for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
	# 	# for c in [1,10]:
	# 		output += 'c = %s\n'%str(c)
	# 		for pl in ['l1','l2']:
	# 			for cw in ['bl', 'nb']:
	# 				mean_tpr, mean_tpr_tr = 0.0, 0.0
	# 				mean_fpr, mean_fpr_tr = np.linspace(0, 1), np.linspace(0, 1)
	# 				for i in range(5):
	# 					if nmf:
	# 						res = pickle.load(open('%s/isg%d/nmfres/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn,isg,nc,s,ntestth,c,pl,cw,i),'r'))
	# 					else:
	# 						res = pickle.load(open('%s/isg%d/res/res_%s_nt%d_c%s_%s_%s_fold%d'%(cdn,isg,s,ntestth,str(c),pl,cw,i),'r'))
	# 					mean_tpr += interp(mean_fpr, res['fpr_te'], res['tpr_te'])
	# 					mean_tpr_tr += interp(mean_fpr_tr, res['fpr_tr'], res['tpr_tr'])
	# 					mean_tpr[0] = 0.0
	# 					mean_tpr_tr[0] = 0.0
	# 				mean_tpr /= 5
	# 				mean_tpr_tr /= 5
	# 				mean_tpr[-1] = 1.0
	# 				mean_tpr_tr[-1] = 1.0
	# 				mean_auc = metrics.auc(mean_fpr, mean_tpr)
	# 				mean_auc_tr = metrics.auc(mean_fpr_tr, mean_tpr_tr)
	# 				# output += '%s %s\n'%(c,pl,cw)
	# 				output += '%s %s\n'%(pl,cw)
	# 				output += '%.03f\ntr_auc:%.03f\n\n'%(mean_auc,mean_auc_tr)
	# 				output += '\n'

	# 				if mean_auc > best_auc:
	# 					best_auc = mean_auc
	# 					best_output = '(c=%s,%s,%s):'%(str(c),pl,cw) + ' %.03f'%mean_auc
	# 				if mean_auc_tr > highest_tr_auc:
	# 					highest_tr_auc = mean_auc_tr
	# 					highest_tr_output = 'highest_tr_auc: (c=%s,%s,%s):'%(str(c),pl,cw) + ' %.03f'%mean_auc_tr
		
	# 	output = best_output + '\n' + highest_tr_output + '\n\n' + output

	# 	if nmf:
	# 		if not os.path.exists('%s/isg%d/output'%(cdn,isg)):
	# 			os.makedirs('%s/isg%d/output'%(cdn,isg))
	# 		if not os.path.exists('%s/isg%d/output/nmfres_%s'%(cdn,isg,s)):
	# 			os.makedirs('%s/isg%d/output/nmfres_%s'%(cdn,isg,s))
	# 		fn = open('%s/isg%d/output/nmfres_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn,isg,s,nc,ntestth),'w')
	# 	else:
	# 		if not os.path.exists('%s/isg%d/output'%(cdn,isg)):
	# 			os.makedirs('%s/isg%d/output'%(cdn,isg))
	# 		if not os.path.exists('%s/isg%d/output/res_%s'%(cdn,isg,s)):
	# 			os.makedirs('%s/isg%d/output/res_%s'%(cdn,isg,s))
	# 		fn = open('%s/isg%d/output/res_%s/auc_nt%d_tuneC.txt'%(cdn,isg,s,ntestth),'w')
	# 	fn.write(output)
	# 	fn.close()

	# 	return best_output, highest_tr_output

	def write_pt_sg_before_filter_pt(ptsg, ptwd, sgs, pt, gt, isg, s, i):
		'''
		Write files.
		'''
		with open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(self.cdn,isg,s,i),'wb') as f:
			pickle.dump(ptsg,f)
		with open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(self.cdn,isg,s,i),'wb') as f:
			pickle.dump(ptwd,f)
		with open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(self.cdn,isg,s,i),'wb') as f:
			pickle.dump(sgs,f)
		with open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(self.cdn,isg,s,i),'wb') as f:
			pickle.dump(pt,f)
		with open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(self.cdn,isg,s,i),'wb') as f:
			pickle.dump(gt,f)

	def moss(fin, fout_prefix, freq_t):
		'''
		Call Java package MoSS.
		'''
		command = 'java'
		mode = '-cp'
		filename = '../moss.jar'
		funcname = 'moss.Miner'
		p1 = '-inelist'
		p2 = '-onelist'
		p3 = '-m1'
		p4 = '-s0.%s'%(freq_t)
		p5 = fin
		p6 = '%s.out'%(fout_prefix)
		p7 = '%s.ids'%(fout_prefix)
		cmd = [command, mode, filename, funcname, p1, p2, p3, p4, p5, p6, p7]
		with open('%s_%s_moss_log'%(fout_prefix,freq_t), 'w') as f:
			subprocess.call(cmd, stderr=f)

	def read_fnout(fnout):
		'''
		read .out file to a hashmap

		e.g. .out
		n 1 mSBP_n_-2
		n 2 mSBP_n_0
		n 3 mSBP_n_0
		e 2 1 tdown
		e 1 3 tup
		g 5
		s 3 2 5 0.0073333136 0 0.0

		hout['n 1 mSBP_n_-2\nn 2 mSBP_n_0\nn 3 mSBP_n_0\ne 2 1 tdown\ne 1 3 tup\n'] = 
			{'gid': 5, 
			'nodeinfo' = n 1 mSBP_n_-2\nn 2 mSBP_n_0\nn 3 mSBP_n_0\ne 2 1 tdown\ne 1 3 tup\ng 5\ns 3 2 5 0.0073333136 0 0.0\n\n'}
		'''
		fn = open(fnout,'r')
		hout = {}; nodestrlist = []; edgestrlist = []; sstr = ''; gstr = ''; gid = 0
		for ln in fn:
			ln = ln.rstrip(" \n")
			if len(ln) == 0:
				continue
			if ln[0] == 'n':
				nodestrlist.append(ln)
			elif ln[0] == 'e':
				edgestrlist.append(ln)
			elif ln[0] == 'g':
				gid = int(ln[2:])
				gstr = ln
			elif ln[0] == 's':
				sstr = ln

				key = ''
				for ns in nodestrlist:
					key += ns + '\n'
				for es in edgestrlist:
					key += es + '\n'

				if key in hout:
					exit('error in read_fnout.')
				hout[key] = {}
				hout[key]['gid'] = gid
				hout[key]['gstr'] = gstr + '\n'
				hout[key]['sstr'] = sstr + '\n\n'
				nodestrlist = []; edgestrlist = []
		return hout

	def read_fnids(fnids):
		'''
		read .ids file to a hashmap hids

		e.g. .ids
		1:5242_45,6675_45,12026_45,25775_45
		5242_45 -> 2 
		6675_45 -> 4 
		6675_45 -> 6 
		12026_45 -> 4 
		12026_45 -> 5 
		12026_45 -> 6 
		25775_45 -> 2 
		25775_45 -> 5 

		hids[1] = '5242_45,6675_45,12026_45,25775_45\n5242_45 -> 2 \n...25775_45 -> 5 \n'
		'''
		fn = open(fnids,'r')
		hids = {}; gid = None; prev_gid = None; sgstr = ''; nodemap = ''; sglist = ''; prev_sglist = ''
		i = 0
		for line in fn:
			i += 1
			m = re.search(r'^(\d+):(.*)$', line)
			if m:
				prev_gid = gid
				prev_sglist = sglist
				gid = int(m.group(1))
				sglist = m.group(2)
				if prev_gid != None:
					sginfo = prev_sglist + '\n' + sgstr
					hids[prev_gid] = sginfo
					nodemap = ''; sgstr = ''
				continue
			m = re.search(r'^([\d_]+) -> (.*) $', line)
			if m:
				nodemap = m.group(1) + ' -> ' + m.group(2)
				sgstr += nodemap + ' \n'

		sginfo = sglist + '\n' + sgstr
		hids[gid] = sginfo

		return hids

	def write_subgraphs(hout, hids, fnout, fnids):
		'''
		Write .out and .ids files.

		fnout = '../data/mimic_m1_s0.%s_fold%d.out'%(freq_t,foldi)
		fnids = '../data/mimic_m1_s0.%s_fold%d.ids'%(freq_t,foldi)
		'''
		fout = open(fnout,'w')
		fids = open(fnids,'w')

		fids.write('id:list\n')
		ks = sorted(hids.keys())
		for k in ks:
			fids.write(str(k)+':')
			fids.write(hids[k])
		fids.close()

		ks = sorted(hout.keys())
		for k in ks:
			fout.write(hout[k])
		fout.close()

	def get_freq_to_trainFreq_map(foldi):
		'''
		Given freq_t list, get corresponding freq_t for training set.
		'''
		print 'fold%d:'%foldi
		ftrnel="%s/mimic_train_fold%d.nel"%(self.cdn,foldi)
		fnel = "%s/mimic_fold%d.nel"%(self.cdn,foldi)
		with open(ftrnel) as f:
			cnt_tr_graphs = sum(1 for _ in f)/nel_graph_length
		with open(fnel) as f:
			cnt_tr_te_graphs = sum(1 for _ in f)/nel_graph_length
		
		l = []

		for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
			freq = math.ceil(float('0.'+freq_t)*cnt_tr_te_graphs/100)
			tr_freq_t = math.floor(math.ceil(cnt_tr_te_graphs*float('0.'+freq_t)/100)*100000/cnt_tr_graphs)*0.001
			tr_freq_t = str(tr_freq_t)[2:]
			tr_freq = math.ceil(float('0.'+tr_freq_t)*cnt_tr_graphs/100)
			l.append(float('0.'+tr_freq_t))
		
		print l

	def directClassfyExperiments(l):
		minp = l[0]
		minc = l[1]
		print minp, minc

		output = ''
		for isg in [0,3]:
			output += 'isg %d:\n'%isg
			for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
			# for freq_t in freq_list_no_med:
				output += 'freq_t %s: '%freq_t
				for foldi in range(5):
					classify(isg,freq_t,2,foldi,minp,minc)
				best_auc, highest_tr_auc = mean_auc(isg, freq_t, 2, nmf=False)
				output += '%s\n%s\n'%(best_auc,highest_tr_auc)
			output += '\n'
		fn = open('../data/mice/directresult_mp%s_mc%s.txt'%(minp,minc),'w')
		fn.write(output)
		fn.close()

	def nmfClassfyExperiments(l):
		minp = l[0]
		minc = l[1]
		print minp, minc

		output = ''
		for isg in [0,3]:
			output += 'isg %d:\n'%isg
			
			cu.checkAndCreate('%s/isg%d/nmf_piks'%(self.cdn,isg))

			for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
				output += 'freq_t %s:\n'%freq_t
				bauc = 0.
				htauc = 0.
				bnc = 0
				for nc in [10,20,30,40,50,60,70,80,90,100,110,120]:
					for foldi in range(5):
						nmfclassify(isg,freq_t,2,foldi,nc=nc,minp=minp,minc=minc)
					best_auc, highest_tr_auc = mean_auc(isg, freq_t, 2, nc=nc, nmf=True)
					if best_auc > bauc:
						bnc = nc
					bauc = max(bauc,best_auc)
					htauc = max(htauc,highest_tr_auc)
				output += 'nc %s: %s\n%s\n'%(bnc,bauc,htauc)
				# output += '\n'
			output += '\n'
		fn = open('../data/mice/nmfresult_mp%s_mc%s.txt'%(minp,minc),'w')
		fn.write(output)
		fn.close()

	def run(feature_type,minp,minc):
		# self.cdn = '../data/mean_last12h'
		# self.cdn = '../data/seed2222/%s/mice/mp%s_mc%s'%(feature_type,minp,minc)
		# print self.cdn
		cu.checkAndCreate(self.cdn)
		for isg in [0,3]:
			cu.checkAndCreate('%s/isg%d'%(self.cdn,isg))
			cu.checkAndCreate('%s/isg%d/pt_sg_w'%(self.cdn,isg))
			cu.checkAndCreate('%s/isg%d/res'%(self.cdn,isg))

		for foldi in range(5):
			
			train = self.ftrain%(self.cdn,foldi)
			test = self.ftest%(self.cdn,foldi)

			ftrnel = "%s/mimic_train_fold%d.nel"%(self.cdn,foldi)
			ftrnode = "%s/mimic_train_fold%d.node"%(self.cdn,foldi)
			fnel = "%s/mimic_fold%d.nel"%(self.cdn,foldi)
			fnode = "%s/mimic_fold%d.node"%(self.cdn,foldi)
			
			interpolation(trcsv=train, tecsv=test, ftrnel=ftrnel, ftrnode=ftrnode, fnel=fnel, fnode=fnode)
			
			get_freq_to_trainFreq_map(foldi)
			
			for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
				subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t=freq_t, foldi=foldi)

				for isg in [0,3]:
					gen_pt_sg_files(isg, freq_t, foldi)

if __name__ == '__main__':
	# clean_data('../data/alldata.csv', '../data/alldata_readmit.csv')
	# cdn = '../data/seed2222'
	# cu.checkAndCreate(cdn)
	# split_nfolds('../alldata_readmit.csv', '../data/seed2222/alldata_readmit', shuffle=True, seed=2222)
	# split_by_feature_type(fn_prefix='%s/alldata_readmit'%cdn, 
	# 	raw_colname=raw_features_for_classify, 
	# 	z_colname=standardized_features_for_classify)
	# split_test_by_patient('../data/seed2222/z')
	train = '../data/seed2222/%s/mice/train_mp%s_mc%s_fold%d.csv'%(feature_type,minp,minc,foldi)
	test = '../data/seed2222/%s/mice/test_mp%s_mc%s_fold%d.csv'%(feature_type,minp,minc,foldi)

	run()

	expri_list = []
	mp_list = ["0.4","0.5","0.65","0"]
	mc_list = ["0.1","0.3","0.4","0.6"]
	for minp in mp_list:
		for minc in mc_list:
			expri_list.append([minp,minc])
	pool = multiprocessing.Pool(16)
	pool.map(nmfClassfyExperiments,expri_list)

