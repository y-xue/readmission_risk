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

# ['readmit', 'sid', 'timeindex',
# 'Antiarrhythmic_agent', 'Anticoagulant', 'Antiplatelet_agent', 
# 'Benzodiazepine', 'beta.Blocking_agent', 'Calcium_channel_blocking_agent', 'Diuretic', 'Hemostatic_agent', 
# 'Inotropic_agent', 'Insulin', 'Nondepolarizing_agent', 'sedatives', 'Somatostatin_preparation', 
# 'Sympathomimetic_agent', 'Thrombolytic_agent', 'Vasodilating_agent', 'AIDS', 'HemMalig', 'MetCarcinoma', 
# 'medtype.label', 'location.label']


# cdn = '../data'
# cdn_diff_imp = '../data/mean'
# cdn_diff_imp = '../data/raw_shuffled'
# cdn_diff_imp = '../data/raw'
# cdn_diff_imp = '../data/test'
# cdn_diff_imp = '../data/no_med_feature/raw_shuffled'
# cdn_diff_imp = '../data/no_med_feature/mean'
# cdn_diff_imp = '../data/mice'
# tcdn = '../data/mean'

# addingcols = 'b3_cols'
# addingcols = 'med_all'

# change cdn in mimic_data.py as well!!!

nel_graph_length = 13

# with med_feature
freq_list = ['001','002','003','004','005','006','008','009', '01','011']
# ['001','002','004','005','007','008', '01','011','013','014']

freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}
# freq_to_trainFreq_map = {'001':'001',
# 						'002':'003',
# 						'004':'005',
# 						'006':'007',
# 						'007':'009',
# 						'009':'011',
# 						'010':'013',
# 						'012':'015',
# 						'014':'017',
# 						'015':'019'}

# without med_feature
freq_list_no_med = ['001','002','004','006','007','009','010', '012','014','015']
# [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019]

def run():
	cdn_diff_imp = '../data/mean_last12h'
	# minp = l[0]
	# minc = l[1]
	# print minp, minc
	# cdn_diff_imp = '../data/mice/mp%s_mc%s'%(str(minp),str(minc))
	print cdn_diff_imp
	if not os.path.exists(cdn_diff_imp):
		os.makedirs(cdn_diff_imp)
	for isg in [0,3]:
		if not os.path.exists('%s/isg%d'%(cdn_diff_imp,isg)):
			os.makedirs('%s/isg%d'%(cdn_diff_imp,isg))
			os.makedirs('%s/isg%d/pt_sg_w'%(cdn_diff_imp,isg))
			os.makedirs('%s/isg%d/res'%(cdn_diff_imp,isg))

	# clean_data('../data/alldata.csv', '../data/alldata_readmit.csv')
	# get_data_in_last12h('../data/alldata_readmit.csv', '../data/alldata_readmit_last12h.csv')
	# split_nfolds('../data/alldata_readmit.csv', '../data/raw_shuffled/alldata_readmit', shuffle=True)
	for nthfold in range(5):
		# for each folds
		# tr = '%s/alldata_readmit_train_fold%d.csv'%(cdn_diff_imp,nthfold)
		# te = '%s/alldata_readmit_test_fold%d.csv'%(cdn_diff_imp,nthfold)
		# tr = '%s/alldata_readmit_train_fold%d_shuffled.csv'%(cdn,nthfold)
		# te = '%s/alldata_readmit_test_fold%d_shuffled.csv'%(cdn,nthfold)
		# tr = '%s/alldata_readmit.csv'%(cdn)
		# tr = '../data/raw_shuffled/alldata_readmit_train_fold%d.csv'%(nthfold)
		# te = '../data/raw_shuffled/alldata_readmit_test_fold%d.csv'%(nthfold)
		# tr = '../data/no_med_feature/raw_shuffled/alldata_readmit_train_fold%d.csv'%(nthfold)
		# te = '../data/no_med_feature/raw_shuffled/alldata_readmit_test_fold%d.csv'%(nthfold)
		
		tr = '%s/alldata_readmit_last12h_train_fold%d.csv'%(cdn_diff_imp,nthfold)
		te = '%s/alldata_readmit_last12h_test_fold%d.csv'%(cdn_diff_imp,nthfold)
		# imputed_tr = '../data/train/last12h_pmm_mp%s_mc%s_fold%d.csv'%(str(minp),str(minc),nthfold)
		# imputed_te = '../data/test/fold%d_mp%s_mc%s.csv'%(nthfold,str(minp),str(minc))

		imputed_tr = '%s/alldata_readmit_last12h_mean_train_fold%d.csv'%(cdn_diff_imp,nthfold)
		imputed_te = '%s/alldata_readmit_last12h_mean_test_fold%d.csv'%(cdn_diff_imp,nthfold)
		impute(tr, te, imputed_tr, imputed_te)

		# interped = '%s/alldata_readmit.csv'%(cdn)
		# interped = '%s/alldata_readmit_fold%d.csv'%(cdn_diff_imp,nthfold)

		ftrnel = "%s/mimic_train_fold%d.nel"%(cdn_diff_imp,nthfold)
		ftrnode = "%s/mimic_train_fold%d.node"%(cdn_diff_imp,nthfold)
		fnel = "%s/mimic_fold%d.nel"%(cdn_diff_imp,nthfold)
		fnode = "%s/mimic_fold%d.node"%(cdn_diff_imp,nthfold)
		# interpolation(trcsv=imputed_tr, tecsv=imputed_te, trtecsv='../data/alldata_readmit_fold%d.csv'%nthfold, 
		# 	ftrnel="../data/mimic_train_fold%d.nel"%nthfold, ftrnode="../data/mimic_train_fold%d.node"%nthfold, 
		# 	fnel="../data/mimic_fold%d.nel"%nthfold, fnode="../data/mimic_fold%d.node"%nthfold)
		interpolation(trcsv=imputed_tr, tecsv=imputed_te, ftrnel=ftrnel, ftrnode=ftrnode, fnel=fnel, fnode=fnode, cdn=cdn_diff_imp)
		# interpolation(trcsv=tr, tecsv=te, ftrnel=ftrnel, ftrnode=ftrnode, fnel=fnel, fnode=fnode)
		
		get_freq_to_trainFreq_map(nthfold,cdn_diff_imp)
		
		for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
		# for freq_t in freq_list_no_med:
		# for freq_t in ['006']:
			subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t=freq_t, nthfold=nthfold, cdn=cdn_diff_imp)

			for isg in [0,3]:
				gen_pt_sg_files(isg, freq_t, nthfold, cdn_diff_imp)
				# for ntestth in [2,5]:
				# for ntestth in [2]:
				# 	classify(isg,freq_t,ntestth,nthfold,1.0)
					# nmfclassify(isg,freq_t,ntestth,nthfold,1,nc=10)
				# 	print_res_auc(isg,freq_t,ntestth,nthfold,1.0)

def clean_data(fn, fout):
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
	# data = data[standardized_features]
	# standardized_features[0] = 'readmit'
	# data.columns = standardized_features
	data.to_csv(fout,index=False)

def get_data_in_last12h(fin, fout):
	data = pd.read_csv(fin)
	last12hData = pd.DataFrame(columns=data.columns)
	gp = data.groupby('sid')
	for sid,g in gp:
		tidx = np.array(g['timeindex'])
		start_tidx = tidx[-1] - 720

		if has_last_12h(tidx):
			last12hData = last12hData.append(g[g['timeindex']>=start_tidx])

	last12hData.to_csv(fout,index=False)

def split_nfolds(fin, fout_prefix, shuffle=False, seed=2222):
	df = pd.read_csv(fin)
	gp = df.groupby('sid')
	kf = KFold(len(gp), n_folds=5, shuffle=shuffle, random_state=seed)
	j = 0
	for train_idx, test_idx in kf:
		trainset = pd.DataFrame(columns=df.columns)
		testset = pd.DataFrame(columns=df.columns)
		i = 0
		for sid,g in gp:
			if i in train_idx:
				trainset = trainset.append(g)
			elif i in test_idx:
				testset = testset.append(g)
			i += 1
		trainset.to_csv('%s_train_fold%d.csv'%(fout_prefix,j),index=False)
		testset.to_csv('%s_test_fold%d.csv'%(fout_prefix,j),index=False)
		j += 1

def impute(ftr, fte, fimptr, fimpte, meth='mean'):
	'''
	if meth=mean, try to impute both alldata and last12hData
	'''
	print 'imputing...'

	if meth == 'mean':
		train = pd.read_csv(ftr)
		test = pd.read_csv(fte)
		mtr = train.fillna(train.mean())
		mte = test.fillna(train.mean())
		mtr.to_csv(fimptr, index=False)
		mte.to_csv(fimpte, index=False)

def interpolation(trcsv, tecsv, ftrnel, ftrnode, fnel, fnode, cdn):
	'''
	trcsv = '../data/alldata_readmit_train_fold%d.csv'%nthfold
	tecsv = '../data/alldata_readmit_test_fold%d.csv'%nthfold
	trtecsv = '../data/alldata_readmit_fold%d.csv'%nthfold
	'''
	print 'interpolating...'
	trtecsv = '%s/trtecsv.csv'%cdn

	x = nggi.NelGraphGenInterpolation()
	x.scan_csv_interpolation(trcsv, ftrnel, ftrnode)

	tr = pd.read_csv(trcsv)
	te = pd.read_csv(tecsv)
	tr = tr.append(te)
	tr.to_csv(trtecsv, index=False)
	x.scan_csv_interpolation(trtecsv, fnel, fnode)

def subgraph_mining(tr_nel, tr_te_nel, freq_t, nthfold, cdn):
	'''
	'''
	print 'subgraph mining...'

	fntr = '%s/mimic_m1_train_fold%d'%(cdn,nthfold)
	fntrte = '%s/mimic_m1_tr_te_fold%d'%(cdn,nthfold)
	# fntr = '../data/testsubgraph/mimic_m1_train_fold%d'%(nthfold)
	# fntrte = '../data/testsubgraph/mimic_m1_tr_te_fold%d'%(nthfold)
	
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

	write_subgraphs(hout, hids, '%s/mimic_m1_fold%d.out'%(cdn,nthfold), '%s/mimic_m1_fold%d.ids'%(cdn,nthfold))
	# write_subgraphs(hout, hids, '../data/testsubgraph/mimic_m1_fold%d.out'%(nthfold), '../data/testsubgraph/mimic_m1_fold%d.ids'%(nthfold))

def gen_pt_sg_files(isg, freq_t, nthfold, cdn_diff_imp):
	print 'gen_pt_sg_files:'
	print isg, freq_t, nthfold
	# mt.pt_sg_w_tensor_gen("%s/mimic_fold%d.nel"%(cdn_diff_imp,nthfold), "%s/mimic_fold%d.node"%(cdn_diff_imp,nthfold), "%s/mimic_m1_s0.%s_fold%d.out"%(cdn_diff_imp,freq_t,nthfold), 
	# 	"%s/mimic_m1_s0.%s_fold%d.ids"%(cdn_diff_imp,freq_t,nthfold), "%s/isg%d/pt_sg_w_%s/mimic.tensor_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), 
	# 	"%s/isg%d/pt_sg_w_%s/mimic_pt_sg.spmat_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), 
	# 	"%s/isg%d/pt_sg_w_%s/mimic_pt_w.spmat_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), 
	# 	"%s/isg%d/pt_sg_w_%s/mimic.ptid_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), 
	# 	"%s/isg%d/pt_sg_w_%s/mimic.sgtid_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), 
	# 	"%s/isg%d/pt_sg_w_%s/mimic.sgstr_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), "%s/isg%d/pt_sg_w_%s/mimic.ntid_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), isg=isg)

	fn_pt_sg_mat = "%s/isg%d/pt_sg_w/mimic_pt_sg.spmat_%s_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold)
	fn_pt_wd_mat = "%s/isg%d/pt_sg_w/mimic_pt_w.spmat_%s_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold)
	fn_sgs = "%s/isg%d/pt_sg_w/mimic.sgstr_%s_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold)
	fn_sparse_tensor = "%s/isg%d/pt_sg_w/mimic.tensor_fold%d"%(cdn_diff_imp,isg,nthfold)
	fn_pt_gt = "%s/isg%d/pt_sg_w/mimic.ptmc_%s_fold%d" % (cdn_diff_imp,isg,freq_t,nthfold)
	fn_pt_lab = "%s/isg%d/pt_sg_w/mimic.ptid_%s_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold)

	mt.pt_sg_w_tensor_gen("%s/mimic_fold%d.nel"%(cdn_diff_imp,nthfold), "%s/mimic_fold%d.node"%(cdn_diff_imp,nthfold), 
		"%s/mimic_m1_fold%d.out"%(cdn_diff_imp,nthfold), "%s/mimic_m1_fold%d.ids"%(cdn_diff_imp,nthfold), 
		fn_sparse_tensor, 
		fn_pt_sg_mat, 
		fn_pt_wd_mat, 
		fn_pt_lab, 
		"%s/isg%d/pt_sg_w/mimic.sgtid_%s_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), 
		fn_sgs, 
		"%s/isg%d/pt_sg_w/mimic.ntid_%s_fold%d"%(cdn_diff_imp,isg,freq_t,nthfold), isg=isg)

	# perform supervised learning
	print 0
	ptsg = md.read_pt_sg_mat(isg,freq_t,nthfold,fn=fn_pt_sg_mat)
	print 1
	ptwd = md.read_pt_wd_mat(isg,freq_t,nthfold,fn=fn_pt_wd_mat)
	print 2
	sgs = md.read_sgs(isg,freq_t,nthfold,fn=fn_sgs)
	print 3
	spt = md.read_sparse_tensor(isg,freq_t,nthfold,fn=fn_sparse_tensor)
	print 4
	(hiso, hsgstr, hsgc, hsgsize) = md.sg_subiso(sgs)
	print 5
	# print ptsg.shape
	(ptsg, sgs, sptsel) = md.filter_sg(ptsg, hiso, sgs, spt=spt)
	# print ptsg.shape
	# df_ptsg = pd.DataFrame(ptsg)
	# df_ptsg.to_csv('%s/isg%d/df_ptsg_%s_fold%d.csv'%(cdn_diff_imp,isg,freq_t,nthfold), index=False)

	print 6
	# ptsg = np.hstack((ptsg, ptwd))
	gt = md.read_pt_gt(isg,freq_t,nthfold,fn=fn_pt_gt)
	print 7
	pt = md.read_pt_lab(isg,freq_t,nthfold,fn=fn_pt_lab)
	print 8
	write_pt_sg_before_filter_pt(ptsg, ptwd, sgs, pt, gt, isg, freq_t, nthfold, cdn_diff_imp)

def classify(isg, s, ntestth, i, minp, minc, cdn_diff_imp):
	print 'classify:'
	print 'minp = %s'%minp, 'minc = %s'%minc
	print 'isg = %d'%isg, 's = %s'%s, 'ntestth = %d'%ntestth, 'fold = %d'%i
	# ptsg = pickle.load(open('%s/isg%d/pt_sg_w_%s/ptsg_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# ptwd = pickle.load(open('%s/isg%d/pt_sg_w_%s/ptwd_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# sgs = pickle.load(open('%s/isg%d/pt_sg_w_%s/sgs_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# pt = pickle.load(open('%s/isg%d/pt_sg_w_%s/pt_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# gt = pickle.load(open('%s/isg%d/pt_sg_w_%s/gt_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	ptwd = pickle.load(open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	sgs = pickle.load(open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	pt = pickle.load(open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	gt = pickle.load(open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# (ptsg, ptwd, pt, gt, sptr) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 5, spt=sptsel)
	(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)


	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te(ptsg, gt, pt)
	(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
		"../data/train/last12h_pmm_mp%s_mc%s_fold%d.csv"%(minp,minc,i), 
		"../data/test/fold%d_mp%s_mc%s.csv"%(i,minp,minc))

	for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
		for pl in ['l1','l2']:
			for cw in ['balanced', None]:
				clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
				# print gt_te.shape
				res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
				if cw == 'balanced':
					cw_abbr = 'bl'
				else:
					cw_abbr = 'nb'
				# with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l1_bl_fold%d'%(cdn_diff_imp,isg,s,ntestth,i),'wb') as f:
				with open('%s/isg%d/res/res_%s_nt%d_c%s_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,str(c),pl,cw_abbr,i),'wb') as f:
					pickle.dump(res,f)

	# clf = LogisticRegression(penalty='l2', class_weight='balanced')
	# res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	# # with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l2_bl_fold%d'%(cdn_diff_imp,isg,s,ntestth,i),'wb') as f:
	# with open('%s/isg%d/res/res_%s_nt%d_l2_bl_fold%d'%(cdn_diff_imp,isg,s,ntestth,i),'wb') as f:
	# 	pickle.dump(res,f)

	# clf = LogisticRegression(penalty='l1', class_weight=None)
	# res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	# # with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l1_nb_fold%d'%(cdn_diff_imp,isg,s,ntestth,i),'wb') as f:
	# with open('%s/isg%d/res/res_%s_nt%d_l1_nb_fold%d'%(cdn_diff_imp,isg,s,ntestth,i),'wb') as f:
	# 	pickle.dump(res,f)

	# clf = LogisticRegression(penalty='l2', class_weight=None)
	# res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	# # with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l2_nb_fold%d'%(cdn_diff_imp,isg,s,ntestth,i),'wb') as f:
	# with open('%s/isg%d/res/res_%s_nt%d_l2_nb_fold%d'%(cdn_diff_imp,isg,s,ntestth,i),'wb') as f:
	# 	pickle.dump(res,f)

def nmfclassify(isg, s, ntestth, i, nc=10, minp='0', minc='0', cdn_diff_imp=None):
	print 'nmfclassify:'
	print '(isg=%d,s=%s,ntestth=%d,fold=%d,nc=%d):'%(isg,s,ntestth,i,nc)
	# ptsg = pickle.load(open('%s/isg%d/pt_sg_w_%s/ptsg_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# ptwd = pickle.load(open('%s/isg%d/pt_sg_w_%s/ptwd_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# sgs = pickle.load(open('%s/isg%d/pt_sg_w_%s/sgs_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# pt = pickle.load(open('%s/isg%d/pt_sg_w_%s/pt_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# gt = pickle.load(open('%s/isg%d/pt_sg_w_%s/gt_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	ptwd = pickle.load(open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	sgs = pickle.load(open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	pt = pickle.load(open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	gt = pickle.load(open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
	# (ptsg, ptwd, pt, gt, sptr) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 5, spt=sptsel)
	(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)

	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, "%s/alldata_readmit_train_fold%d.csv"%(cdn_diff_imp,i), 
	# 	"%s/alldata_readmit_test_fold%d.csv"%(cdn_diff_imp,i))
	
	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
	# 	"%s/alldata_readmit_train_fold%d.csv"%(cdn_diff_imp,i), 
	# 	"%s/alldata_readmit_test_fold%d.csv"%(cdn_diff_imp,i),
	# 	medfntr="%s/alldata_readmit_mean_train_fold%d_age+medtype_lastMeasures.csv"%(cdn_diff_imp,i),
	# 	medfnte="%s/alldata_readmit_mean_test_fold%d_age+medtype_lastMeasures.csv"%(cdn_diff_imp,i))
	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
	# 	"%s/alldata_readmit_train_fold%d.csv"%(cdn_diff_imp,i), 
	# 	"%s/alldata_readmit_test_fold%d.csv"%(cdn_diff_imp,i),
	# 	medfntr="%s/alldata_readmit_mean_train_fold%d_med_all_lastMeasures.csv"%(cdn_diff_imp,i),
	# 	medfnte="%s/alldata_readmit_mean_test_fold%d_med_all_lastMeasures.csv"%(cdn_diff_imp,i))
	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te, filtered_tridlist, filtered_teidlist) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
	# 	"%s/alldata_readmit_train_fold%d.csv"%(cdn_diff_imp,i), 
	# 	"%s/alldata_readmit_test_fold%d.csv"%(cdn_diff_imp,i))#,
		# medfntr="%s/alldata_readmit_mean_train_fold%d_%s_lastMeasures.csv"%(cdn_diff_imp,i,addingcols),
		# medfnte="%s/alldata_readmit_mean_test_fold%d_%s_lastMeasures.csv"%(cdn_diff_imp,i,addingcols))
	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
	# 	"%s/alldata_readmit_train_fold%d.csv"%(cdn_diff_imp,i), 
	# 	"%s/alldata_readmit_test_fold%d.csv"%(cdn_diff_imp,i))
	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
	# 	"../data/train/last12h_pmm_mp%s_mc%s_fold%d.csv"%(minp,minc,i),
	# 	"../data/test/fold%d_mp%s_mc%s.csv"%(i,minp,minc))
	(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
		'%s/alldata_readmit_last12h_mean_train_fold%d.csv'%(cdn_diff_imp,i),
		'%s/alldata_readmit_last12h_mean_test_fold%d.csv'%(cdn_diff_imp,i))

	for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
		for pl in ['l1','l2']:
			for cw in ['balanced', None]:
				clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
				
				(m, clf, res) = mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
					fnfmt='%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(cdn_diff_imp,isg,s,i,nc), nc=nc, clf=clf)
					# medfntr="%s/alldata_readmit_mean_train_fold%d_%s_lastMeasures.csv"%(cdn_diff_imp,i,addingcols), 
					# medfnte="%s/alldata_readmit_mean_test_fold%d_%s_lastMeasures.csv"%(cdn_diff_imp,i,addingcols),
					# filtered_tridlist=filtered_tridlist, filtered_teidlist=filtered_teidlist)

				if cw == 'balanced':
					cw_abbr = 'bl'
				else:
					cw_abbr = 'nb'

				# if not os.path.exists('%s/isg%d/nmfres/%d'%(cdn_diff_imp,isg,nc)):
				# 	os.makedirs('%s/isg%d/nmfres/%d'%(cdn_diff_imp,isg,nc))
				# with open('%s/isg%d/nmfres/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw_abbr,i),'wb') as f:
				# 	pickle.dump(res,f)
				# if not os.path.exists('%s/isg%d/nmfres_best6med/%d'%(cdn_diff_imp,isg,nc)):
				# 	os.makedirs('%s/isg%d/nmfres_best6med/%d'%(cdn_diff_imp,isg,nc))
				# with open('%s/isg%d/nmfres_best6med/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw_abbr,i),'wb') as f:
				# 	pickle.dump(res,f)
				# if not os.path.exists('%s/isg%d/nmfres_allmed/%d'%(cdn_diff_imp,isg,nc)):
				# 	os.makedirs('%s/isg%d/nmfres_allmed/%d'%(cdn_diff_imp,isg,nc))
				# with open('%s/isg%d/nmfres_allmed/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw_abbr,i),'wb') as f:
				# 	pickle.dump(res,f)
				# if not os.path.exists('%s/isg%d/nmfres_age+medtype/%d'%(cdn_diff_imp,isg,nc)):
				# 	os.makedirs('%s/isg%d/nmfres_age+medtype/%d'%(cdn_diff_imp,isg,nc))
				# with open('%s/isg%d/nmfres_age+medtype/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw_abbr,i),'wb') as f:
				# 	pickle.dump(res,f)
				# if not os.path.exists('%s/isg%d/nmfres_%s/%d'%(cdn_diff_imp,isg,addingcols,nc)):
				# 	os.makedirs('%s/isg%d/nmfres_%s/%d'%(cdn_diff_imp,isg,addingcols,nc))
				# with open('%s/isg%d/nmfres_%s/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,addingcols,nc,s,ntestth,c,pl,cw_abbr,i),'wb') as f:
				# 	pickle.dump(res,f)
				if not os.path.exists('%s/isg%d/nmfres/%d'%(cdn_diff_imp,isg,nc)):
					os.makedirs('%s/isg%d/nmfres/%d'%(cdn_diff_imp,isg,nc))
				with open('%s/isg%d/nmfres/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw_abbr,i),'wb') as f:
					pickle.dump(res,f)

def print_res_auc(isg, s, ntestth, i, c, cdn_diff_imp):
	output = ''
	print '(isg=%d,s=%s,ntestth=%d,fold=%d,c=%f):'%(isg,s,ntestth,i,c)
	for pl in ['l1','l2']:
		for cw in ['bl', 'nb']:
			# res = pickle.load(open('%s/isg%d/pt_sg_w_%s/res_nt%d_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,pl,cw,i),'r'))
			res = pickle.load(open('%s/isg%d/res/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,c,pl,cw,i),'r'))
			print pl,cw
			print res['auc_te']
			print 
			output += '%s %s\n'%(pl,cw)
			output += '%.03f\n\n'%res['auc_te']
	if not os.path.exists('%s/isg%d/res_%s'%(cdn_diff_imp,isg,s)):
		os.makedirs('%s/isg%d/res_%s'%(cdn_diff_imp,isg,s))
	fn = open('%s/isg%d/res_%s/fold%d_nt%d_c%f.txt'%(cdn_diff_imp,isg,s,i,ntestth,c),'w')
	fn.write(output)
	fn.close()

# def nmf_print_res_auc(isg, s, ntestth, i):
# 	output = ''
# 	print '(isg=%d,s=%s,ntestth=%d,fold=%d):'%(isg,s,ntestth,i)
# 	for pl in ['l1','l2']:
# 		for cw in ['bl', 'nb']:
# 			# res = pickle.load(open('%s/isg%d/pt_sg_w_%s/res_nt%d_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,pl,cw,i),'r'))
# 			res = pickle.load(open('%s/isg%d/nmfres/res_%s_nt%d_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,pl,cw,i),'r'))
# 			print pl,cw
# 			print res['auc_te']
# 			print 
# 			output += '%s %s\n'%(pl,cw)
# 			output += '%.03f\n\n'%res['auc_te']
# 	if not os.path.exists('%s/isg%d/nmfres_%s'%(cdn_diff_imp,isg,s)):
# 		os.makedirs('%s/isg%d/nmfres_%s'%(cdn_diff_imp,isg,s))
# 	fn = open('%s/isg%d/nmfres_%s/fold%d_nt%d.txt'%(cdn_diff_imp,isg,s,i,ntestth),'w')
# 	fn.write(output)
# 	fn.close()

def mean_auc(isg, s, ntestth, nc=10, nmf=False, cdn_diff_imp=None):
	print '(isg=%d,s=%s,ntestth=%d):'%(isg,s,ntestth)
	output = ''
	best_auc = 0
	best_output = ''

	highest_tr_auc = 0
	highest_tr_output = ''
	for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
	# for c in [1,10]:
		output += 'c = %s\n'%str(c)
		for pl in ['l1','l2']:
			for cw in ['bl', 'nb']:
				mean_tpr, mean_tpr_tr = 0.0, 0.0
				mean_fpr, mean_fpr_tr = np.linspace(0, 1), np.linspace(0, 1)
				for i in range(5):
					# res = pickle.load(open('%s/isg%d/pt_sg_w_%s/res_nt%d_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,pl,cw,i),'r'))
					if nmf:
						# res = pickle.load(open('%s/isg%d/nmfres/res_%s_nt%d_%s_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,c,pl,cw,i),'r'))
						# res = pickle.load(open('%s/isg%d/nmfres_best6med/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw,i),'r'))
						# res = pickle.load(open('%s/isg%d/nmfres/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw,i),'r'))
						# res = pickle.load(open('%s/isg%d/nmfres_allmed/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw,i),'r'))
						# res = pickle.load(open('%s/isg%d/nmfres_age+medtype/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw,i),'r'))
						# res = pickle.load(open('%s/isg%d/nmfres_%s/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,addingcols,nc,s,ntestth,c,pl,cw,i),'r'))
						res = pickle.load(open('%s/isg%d/nmfres/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(cdn_diff_imp,isg,nc,s,ntestth,c,pl,cw,i),'r'))
					else:
						# res_01_nt2_c1.000000_l1_bl_fold1
						# res = pickle.load(open('%s/isg%d/res/res_%s_nt%d_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,pl,cw,i),'r'))
						res = pickle.load(open('%s/isg%d/res/res_%s_nt%d_c%s_%s_%s_fold%d'%(cdn_diff_imp,isg,s,ntestth,str(c),pl,cw,i),'r'))
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
				# output += '%s %s\n'%(c,pl,cw)
				output += '%s %s\n'%(pl,cw)
				output += '%.03f\ntr_auc:%.03f\n\n'%(mean_auc,mean_auc_tr)
				output += '\n'

				if mean_auc > best_auc:
					best_auc = mean_auc
					best_output = '(c=%s,%s,%s):'%(str(c),pl,cw) + ' %.03f'%mean_auc
				if mean_auc_tr > highest_tr_auc:
					highest_tr_auc = mean_auc_tr
					highest_tr_output = 'highest_tr_auc: (c=%s,%s,%s):'%(str(c),pl,cw) + ' %.03f'%mean_auc_tr
	
	output = best_output + '\n' + highest_tr_output + '\n\n' + output

	if nmf:
		# if not os.path.exists('%s/isg%d/nmfres_%s'%(cdn_diff_imp,isg,s)):
		# 	os.makedirs('%s/isg%d/nmfres_%s'%(cdn_diff_imp,isg,s))
		# fn = open('%s/isg%d/nmfres_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,s,nc,ntestth),'w')
		# if not os.path.exists('%s/isg%d/nmfres_best6med_%s'%(cdn_diff_imp,isg,s)):
		# 	os.makedirs('%s/isg%d/nmfres_best6med_%s'%(cdn_diff_imp,isg,s))
		# fn = open('%s/isg%d/nmfres_best6med_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,s,nc,ntestth),'w')
		# if not os.path.exists('%s/isg%d/nmfres_allmed_%s'%(cdn_diff_imp,isg,s)):
		# 	os.makedirs('%s/isg%d/nmfres_allmed_%s'%(cdn_diff_imp,isg,s))
		# fn = open('%s/isg%d/nmfres_allmed_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,s,nc,ntestth),'w')
		# if not os.path.exists('%s/isg%d/nmfres_nmfres_age+medtype_%s'%(cdn_diff_imp,isg,s)):
		# 	os.makedirs('%s/isg%d/nmfres_nmfres_age+medtype_%s'%(cdn_diff_imp,isg,s))
		# fn = open('%s/isg%d/nmfres_nmfres_age+medtype_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,s,nc,ntestth),'w')
		# if not os.path.exists('%s/isg%d/nmfres_%s_%s'%(cdn_diff_imp,isg,addingcols,s)):
		# 	os.makedirs('%s/isg%d/nmfres_%s_%s'%(cdn_diff_imp,isg,addingcols,s))
		# fn = open('%s/isg%d/nmfres_%s_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,addingcols,s,nc,ntestth),'w')
		if not os.path.exists('%s/isg%d/output'%(cdn_diff_imp,isg)):
			os.makedirs('%s/isg%d/output'%(cdn_diff_imp,isg))
		if not os.path.exists('%s/isg%d/output/nmfres_%s'%(cdn_diff_imp,isg,s)):
			os.makedirs('%s/isg%d/output/nmfres_%s'%(cdn_diff_imp,isg,s))
		fn = open('%s/isg%d/output/nmfres_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,s,nc,ntestth),'w')
	else:
		if not os.path.exists('%s/isg%d/output'%(cdn_diff_imp,isg)):
			os.makedirs('%s/isg%d/output'%(cdn_diff_imp,isg))
		if not os.path.exists('%s/isg%d/output/res_%s'%(cdn_diff_imp,isg,s)):
			os.makedirs('%s/isg%d/output/res_%s'%(cdn_diff_imp,isg,s))
		fn = open('%s/isg%d/output/res_%s/auc_nt%d_tuneC.txt'%(cdn_diff_imp,isg,s,ntestth),'w')
	fn.write(output)
	fn.close()

	return best_output, highest_tr_output

def write_pt_sg_before_filter_pt(ptsg, ptwd, sgs, pt, gt, isg, s, i, cdn_diff_imp):
	# with open('%s/isg%d/pt_sg_w_%s/ptsg_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
	# 	pickle.dump(ptsg,f)
	# with open('%s/isg%d/pt_sg_w_%s/ptwd_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
	# 	pickle.dump(ptwd,f)
	# with open('%s/isg%d/pt_sg_w_%s/sgs_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
	# 	pickle.dump(sgs,f)
	# with open('%s/isg%d/pt_sg_w_%s/pt_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
	# 	pickle.dump(pt,f)
	# with open('%s/isg%d/pt_sg_w_%s/gt_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
	# 	pickle.dump(gt,f)

	with open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
		pickle.dump(ptsg,f)
	with open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
		pickle.dump(ptwd,f)
	with open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
		pickle.dump(sgs,f)
	with open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
		pickle.dump(pt,f)
	with open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(cdn_diff_imp,isg,s,i),'wb') as f:
		pickle.dump(gt,f)

def has_last_12h(tidx):
	if len(tidx) > 1 and tidx[-1] >= 720:
		return True
	else:
		return False

# def combine_tr_te(ftrx, ftex, ftry):
# 	for i in range(5):
# 		xtrain = pd.read_csv('../data/train/alldata_last12h_X_train_fold0.csv')
# 		ytrain = pd.Series.from_csv('../data/train/alldata_last12h_y_train_fold0.csv')
# 		xtest = pd.read_csv('../data/test/alldata_last12h_X_test_fold0.csv')
# 		ytest = pd.Series.from_csv('../data/test/alldata_last12h_y_test_fold0.csv')

# 		xtrain['died'] = ytrain
# 		xtest['died'] = ytest

# 		cols = xtrain.columns.tolist()
# 		cols = cols[-1:] + cols[:-1]
# 		xtrain = xtrain[cols]
# 		xtest = xtest[cols]

# 		# data = xtrain.append(xtest)
# 		print data.columns

# 		data.to_csv('../data/alldata_last12h_mean_fold%d.csv'%i,index=False)

def moss(fin, fout_prefix, freq_t):
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
	fnout = '../data/mimic_m1_s0.%s_fold%d.out'%(freq_t,nthfold)
	fnids = '../data/mimic_m1_s0.%s_fold%d.ids'%(freq_t,nthfold)
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

def test_moss(freq_t,nthfold):
	ftrnel = "%s/mimic_train_fold%d.nel"%(cdn_diff_imp,nthfold)
	fnel = "%s/mimic_fold%d.nel"%(cdn_diff_imp,nthfold)
	if not os.path.exists('%s/atestmoss/fold%d'%(cdn_diff_imp,nthfold)):
		os.makedirs('%s/atestmoss/fold%d'%(cdn_diff_imp,nthfold))
	fntr = '%s/atestmoss/fold%d/mimic_m1_train_%s'%(cdn_diff_imp,nthfold,freq_t)
	fntrte = '%s/atestmoss/fold%d/mimic_m1_tr_te_%s'%(cdn_diff_imp,nthfold,freq_t)
	
	tr_freq_t = freq_to_trainFreq_map[freq_t]

	moss(ftrnel, fntr, tr_freq_t)
	moss(fnel, fntrte, freq_t)

def get_freq_to_trainFreq_map(nthfold, cdn):
	print 'fold%d:'%nthfold
	ftrnel="%s/mimic_train_fold%d.nel"%(cdn,nthfold)
	fnel = "%s/mimic_fold%d.nel"%(cdn,nthfold)
	with open(ftrnel) as f:
		cnt_tr_graphs = sum(1 for _ in f)/nel_graph_length
	with open(fnel) as f:
		cnt_tr_te_graphs = sum(1 for _ in f)/nel_graph_length
	# print 'train graphs: %d'%cnt_tr_graphs
	# print 'all graphs: %d'%cnt_tr_te_graphs
	l = []

	for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
	# for freq_t in freq_list_no_med:
		# print 'freq_t = %s'%freq_t
		freq = math.ceil(float('0.'+freq_t)*cnt_tr_te_graphs/100)
		# print 'freq = %f'%(float('0.'+freq_t)*cnt_tr_te_graphs/100)
		tr_freq_t = math.floor(math.ceil(cnt_tr_te_graphs*float('0.'+freq_t)/100)*100000/cnt_tr_graphs)*0.001
		tr_freq_t = str(tr_freq_t)[2:]
		# print 'tr_freq_t = %s'%tr_freq_t
		# print 'tr_freq = %f'%(float('0.'+tr_freq_t)*cnt_tr_graphs/100)
		tr_freq = math.ceil(float('0.'+tr_freq_t)*cnt_tr_graphs/100)
		l.append(float('0.'+tr_freq_t))
	print l

def directClassfyExperiments(l):
	# for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
	# 	for isg in [0,3]:
	# 		for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
	# 			for nthfold in range(5):
	# 				classify(isg,freq_t,2,nthfold,c)
	# 				print_res_auc(isg,freq_t,2,nthfold,c)
	# 		mean_auc(isg, freq_t, 2, nc=nc, nmf=False)
	minp = l[0]
	minc = l[1]
	print minp, minc
	cdn_diff_imp = '../data/mice/mp%s_mc%s'%(str(minp),str(minc))
	print cdn_diff_imp

	output = ''
	for isg in [0,3]:
		output += 'isg %d:\n'%isg
		for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
		# for freq_t in freq_list_no_med:
			output += 'freq_t %s: '%freq_t
			for nthfold in range(5):
				classify(isg,freq_t,2,nthfold,minp,minc,cdn_diff_imp)
			best_auc, highest_tr_auc = mean_auc(isg, freq_t, 2, nmf=False, cdn_diff_imp=cdn_diff_imp)
			output += '%s\n%s\n'%(best_auc,highest_tr_auc)
		output += '\n'
	fn = open('../data/mice/directresult_mp%s_mc%s.txt'%(minp,minc),'w')
	fn.write(output)
	fn.close()

def nmfClassfyExperiments(l):
	minp = l[0]
	minc = l[1]
	print minp, minc
	cdn_diff_imp = '../data/mice/mp%s_mc%s'%(str(minp),str(minc))
	print cdn_diff_imp

	output = ''
	for isg in [0,3]:
		output += 'isg %d:\n'%isg
		
		if not os.path.exists('%s/isg%d/nmf_piks'%(cdn_diff_imp,isg)):
			os.makedirs('%s/isg%d/nmf_piks'%(cdn_diff_imp,isg))

		for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
		# for freq_t in freq_list_no_med:
			output += 'freq_t %s:\n'%freq_t
			# for nc in [30,50,100]:
			bauc = 0.
			htauc = 0.
			for nc in [10,20,30,40,50,60,70,80,90,100,110,120]:
				output += 'nc %d: '%nc
				for nthfold in range(5):
					nmfclassify(isg,freq_t,2,nthfold,nc=nc,minp=minp,minc=minc,cdn_diff_imp=cdn_diff_imp)
				best_auc, highest_tr_auc = mean_auc(isg, freq_t, 2, nc=nc, nmf=True, cdn_diff_imp=cdn_diff_imp)
				bauc = max(bauc,best_auc)
				htauc = max(htauc,highest_tr_auc)
			output += '%s\n%s\n'%(bauc,htauc)
			# output += '\n'
		output += '\n'
	# fn = open('%s/nmfresult.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_allmed.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_best6med.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_%s.txt'%(cdn_diff_imp,addingcols),'w')
	fn = open('../data/mice/nmfresult_mp%s_mc%s.txt'%(minp,minc),'w')
	fn.write(output)
	fn.close()

def get_lastMeasures(fn, cols):
	# cols = ['sid', 'timeindex', 'Age.standardized', 'Antiarrhythmic_agent', 'Anticoagulant', 'Antiplatelet_agent', 
	# 'Benzodiazepine', 'beta.Blocking_agent', 'Calcium_channel_blocking_agent', 'Diuretic', 'Hemostatic_agent', 
	# 'Inotropic_agent', 'Insulin', 'Nondepolarizing_agent', 'sedatives', 'Somatostatin_preparation', 
	# 'Sympathomimetic_agent', 'Thrombolytic_agent', 'Vasodilating_agent', 'AIDS', 'HemMalig', 'MetCarcinoma', 
	# 'medtype.label']
	# cols = ['sid', 'timeindex', 'Anticoagulant', 'Somatostatin_preparation', 
	# 'Sympathomimetic_agent', 'Vasodilating_agent', 'AIDS', 'MetCarcinoma']
	# cols = ['sid', 'timeindex', 'Age.standardized', 'medtype.label']
	data = pd.read_csv('%s.csv'%fn)
	data = data[cols]
	grouped = data.groupby('sid')

	lm = pd.DataFrame(columns=cols)
	for sid, group in grouped:
		t = group.sort_values(by = 'timeindex', ascending = False)
		s = pd.Series(index=cols)
		for col in t:
			for i,v in t[col].iteritems():
				if np.isnan(v):
					continue
				else:
					s[col] = v
					break
		lm = lm.append(s,ignore_index=True)

	lm.to_csv('%s_%s_lastMeasures.csv'%(fn,addingcols),index=False)
	# lm.to_csv('%s_med_all_lastMeasures.csv'%fn,index=False)

def rfe(rfe_n):
	isg = 3
	f1 = '002'
	nc1 = 60
	c1 = 10
	p1 = 'l1'
	cw1 = None
	
	f2 = '004'
	nc2 = 100
	c2 = 50
	p2 = 'l2'
	cw2 = None
	# rfe_n = 30
	for nthfold in range(5):
		nmfclassify_once(isg,f1,2,nthfold,c=c1,nc=nc1,pl=p1,cw=cw1,rfe_n=rfe_n)
	mean_auc_once(isg,f1,2,c=c1,nc=nc1,pl=p1,cw=cw1,nmf=True,rfe_n=rfe_n)

	for nthfold in range(5):
		nmfclassify_once(isg,f2,2,nthfold,c=c2,nc=nc2,pl=p2,cw=cw2,rfe_n=rfe_n)
	mean_auc_once(isg,f2,2,c=c2,nc=nc2,pl=p2,cw=cw2,nmf=True,rfe_n=rfe_n)

def nmfclassify_once(isg, s, ntestth, i, nc=10, c=1.0, pl='l1', cw=None, rfe_n=30):
	print 'nmfclassify:'
	print '(isg=%d,s=%s,ntestth=%d,fold=%d,nc=%d):'%(isg,s,ntestth,i,nc)
	ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(tcdn,isg,s,i),'rb'))
	ptwd = pickle.load(open('%s/isg%d/pt_sg_w/ptwd_%s_fold%d'%(tcdn,isg,s,i),'rb'))
	sgs = pickle.load(open('%s/isg%d/pt_sg_w/sgs_%s_fold%d'%(tcdn,isg,s,i),'rb'))
	pt = pickle.load(open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(tcdn,isg,s,i),'rb'))
	gt = pickle.load(open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(tcdn,isg,s,i),'rb'))
	# (ptsg, ptwd, pt, gt, sptr) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 5, spt=sptsel)
	(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)
	
	(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
		"%s/alldata_readmit_train_fold%d.csv"%(tcdn,i), 
		"%s/alldata_readmit_test_fold%d.csv"%(tcdn,i))
	clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
	
	(m, clf, res) = mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
		fnfmt='%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(tcdn,isg,s,i,nc), nc=nc, clf=clf, rfe=True, rfe_n=rfe_n)

	if cw == 'balanced':
		cw_abbr = 'bl'
	else:
		cw_abbr = 'nb'

	if not os.path.exists('%s/isg%d/nmfres_rfe/%d'%(tcdn,isg,nc)):
		os.makedirs('%s/isg%d/nmfres_rfe/%d'%(tcdn,isg,nc))
	with open('%s/isg%d/nmfres_rfe/%d/res_%s_nt%d_c%f_%s_%s_fold%d_rfe%d'%(tcdn,isg,nc,s,ntestth,c,pl,cw_abbr,i,rfe_n),'wb') as f:
		pickle.dump(res,f)
	# if not os.path.exists('%s/isg%d/nmfres_rfe_t/%d'%(tcdn,isg,nc)):
	# 	os.makedirs('%s/isg%d/nmfres_rfe_t/%d'%(tcdn,isg,nc))
	# with open('%s/isg%d/nmfres_rfe_t/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(tcdn,isg,nc,s,ntestth,c,pl,cw_abbr,i),'wb') as f:
	# 	pickle.dump(res,f)

def mean_auc_once(isg, s, ntestth, c=1.0, nc=10 ,pl='l1' ,cw=None, nmf=False, rfe_n=30):
	print '(isg=%d,s=%s,ntestth=%d):'%(isg,s,ntestth)
	if cw == 'balanced':
		cw_abbr = 'bl'
	else:
		cw_abbr = 'nb'
	output = ''
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1)
	for i in range(5):
		if nmf:
			res = pickle.load(open('%s/isg%d/nmfres_rfe/%d/res_%s_nt%d_c%f_%s_%s_fold%d_rfe%d'%(tcdn,isg,nc,s,ntestth,c,pl,cw_abbr,i,rfe_n),'r'))
			# res = pickle.load(open('%s/isg%d/nmfres_rfe_t/%d/res_%s_nt%d_c%f_%s_%s_fold%d'%(tcdn,isg,nc,s,ntestth,c,pl,cw_abbr,i),'r'))
		else:
			res = pickle.load(open('%s/isg%d/res/res_%s_nt%d_c%f_%s_%s_fold%d'%(tcdn,isg,s,ntestth,c,pl,cw_abbr,i),'r'))
		mean_tpr += interp(mean_fpr, res['fpr_te'], res['tpr_te'])
		mean_tpr[0] = 0.0
	mean_tpr /= 5
	mean_tpr[-1] = 1.0
	mean_auc = metrics.auc(mean_fpr, mean_tpr)
	output += '%s %s\n'%(pl,cw_abbr)
	output += '%.03f\n'%(mean_auc)
	output += '\n'

	if nmf:
		if not os.path.exists('%s/isg%d/nmfres_%s_rfe'%(tcdn,isg,s)):
			os.makedirs('%s/isg%d/nmfres_%s_rfe'%(tcdn,isg,s))
		fn = open('%s/isg%d/nmfres_%s_rfe/auc_nc%d_nt%d_rfe%d_tuneC.txt'%(tcdn,isg,s,nc,ntestth,rfe_n),'w')
		# fn = open('%s/isg%d/nmfres_%s_rfe_t/auc_nc%d_nt%d_tuneC.txt'%(tcdn,isg,s,nc,ntestth),'w')
	else:
		if not os.path.exists('%s/isg%d/res_%s'%(tcdn,isg,s)):
			os.makedirs('%s/isg%d/res_%s'%(tcdn,isg,s))
		fn = open('%s/isg%d/res_%s/auc_nt%d_tuneC.txt'%(tcdn,isg,s,ntestth),'w')
	fn.write(output)
	fn.close()

def get_nmf_res(l):
	minp = l[0]
	minc = l[1]
	print minp, minc
	cdn_diff_imp = '../data/mice/mp%s_mc%s'%(str(minp),str(minc))
	print cdn_diff_imp

	output = ''
	for isg in [0,3]:
		output += 'isg %d:\n'%isg
		
		# if not os.path.exists('%s/isg%d/nmf_piks'%(cdn_diff_imp,isg)):
		# 	os.makedirs('%s/isg%d/nmf_piks'%(cdn_diff_imp,isg))

		for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
		# for freq_t in freq_list_no_med:
			output += 'freq_t %s:\n'%freq_t
			best_auc = 0
			highest_tr_auc = 0
			# for nc in [30,50,100]:
			for nc in [10,20,30,40,50,60,70,80,90,100,110,120]:
				# output += 'nc %d: '%nc
				# for nthfold in range(5):
				# 	nmfclassify(isg,freq_t,2,nthfold,nc=nc,minp=minp,minc=minc,cdn_diff_imp=cdn_diff_imp)
				if os.path.isfile('%s/isg%d/output/nmfres_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,freq_t,nc,2)):
					fn = open('%s/isg%d/output/nmfres_%s/auc_nc%d_nt%d_tuneC.txt'%(cdn_diff_imp,isg,freq_t,nc,2),'r')
					ln_cnt = 0
					for ln in fn:
						if ln_cnt >= 2:
							break
						m = re.search(r'(\(.*\):) (.*)',ln)
						if m:
							if ln_cnt == 0:
								best_auc = max(best_auc, float(m.group(2)))
							elif ln_cnt == 1:
								highest_tr_auc = max(highest_tr_auc, float(m.group(2)))
							if minp == '0.4' and minc == '0.1' and isg == 0 and freq_t == '001' and nc == 100:
								print float(m.group(2))
							ln_cnt += 1
						# m = re.search(r'highest_tr_auc: (\(.*\):) (.*)', ln)
						# if m:
						# 	highest_tr_auc = max(highest_tr_auc, float(m.group(2)))
						# 	if minp == '0.4' and minc == '0.1' and isg == 0 and freq_t == '001' and nc == 100:
						# 		print float(m.group(2))
						# 	break
				# best_auc, highest_tr_auc = mean_auc(isg, freq_t, 2, nc=nc, nmf=True, cdn_diff_imp=cdn_diff_imp)
			output += '%s\n%s\n'%(best_auc,highest_tr_auc)
			# output += '\n'
		output += '\n'
	# fn = open('%s/nmfresult.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_allmed.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_best6med.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_%s.txt'%(cdn_diff_imp,addingcols),'w')
	fn = open('../data/mice/nmfresult_tmp_11-16/nmfresult_mp%s_mc%s_tmp_11-16.txt'%(minp,minc),'w')
	fn.write(output)
	fn.close()
# ()
# (c=0.5,l1,nb): 0.579
# highest_tr_auc: (c=50,l2,bl): 0.738

def nmfClassfyExperiments_mean():
	cdn_diff_imp = '../data/mean_last12h'
	print cdn_diff_imp

	output = ''
	for isg in [0,3]:
		output += 'isg %d:\n'%isg
		
		if not os.path.exists('%s/isg%d/nmf_piks'%(cdn_diff_imp,isg)):
			os.makedirs('%s/isg%d/nmf_piks'%(cdn_diff_imp,isg))

		for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
		# for freq_t in freq_list_no_med:
			output += 'freq_t %s:\n'%freq_t
			# for nc in [30,50,100]:
			bauc = 0.
			htauc = 0.
			for nc in [10,20,30,40,50,60,70,80,90,100,110,120]:
				output += 'nc %d: '%nc
				for nthfold in range(5):
					nmfclassify(isg,freq_t,2,nthfold,nc=nc,cdn_diff_imp=cdn_diff_imp)
				best_auc, highest_tr_auc = mean_auc(isg, freq_t, 2, nc=nc, nmf=True, cdn_diff_imp=cdn_diff_imp)
				bauc = max(bauc,best_auc)
				htauc = max(htauc,highest_tr_auc)
			output += '%s\n%s\n'%(bauc,htauc)
			# output += '\n'
		output += '\n'
	# fn = open('%s/nmfresult.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_allmed.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_best6med.txt'%cdn_diff_imp,'w')
	# fn = open('%s/nmfresult_%s.txt'%(cdn_diff_imp,addingcols),'w')
	fn = open('%s/nmfresult.txt'%(cdn_diff_imp),'w')
	fn.write(output)
	fn.close()

def split_by_feature_type(fn_prefix, raw_colname, z_colname):
	for i in range(5):
		train = pd.read_csv('%s_train_fold%d.csv'%(fn_prefix,i))
		test = pd.read_csv('%s_test_fold%d.csv'%(fn_prefix,i))
		raw_train = train[raw_colname]
		raw_test = test[raw_colname]
		z_train = train[z_colname]
		z_test = test[z_colname]
		cu.checkAndCreate('%s/raw/'%cdn)
		cu.checkAndCreate('%s/z/'%cdn)
		raw_train.to_csv('%s/raw/train_fold%d.csv'%(cdn,i),index=False)
		raw_test.to_csv('%s/raw/test_fold%d.csv'%(cdn,i),index=False)
		z_train.to_csv('%s/z/train_fold%d.csv'%(cdn,i),index=False)
		z_test.to_csv('%s/z/test_fold%d.csv'%(cdn,i),index=False)

def split_test_by_patient(tcdn):
	for i in range(5):
		test = pd.read_csv('%s/test_fold%d.csv'%(tcdn,i))
		gp = test.groupby('sid')
		cu.checkAndCreate('%s/mice/imputing/test_fold%d'%(tcdn,i))
		fn = open('%s/mice/imputing/test_fold%d/sid_list.txt'%(tcdn,i),'w')
		for sid, group in gp:
			group.to_csv('%s/mice/imputing/test_fold%d/%d.csv'%(tcdn,i,sid),index=False)
			fn.write('%d\n'%sid)
		fn.close()

if __name__ == '__main__':
	cdn = '../data'
	# clean_data('../data/alldata.csv', '../data/alldata_readmit.csv')
	cdn = '../data/seed2222'
	cu.checkAndCreate(cdn)
	# split_nfolds('../alldata_readmit.csv', '../data/seed2222/alldata_readmit', shuffle=True, seed=2222)
	# split_by_feature_type(fn_prefix='%s/alldata_readmit'%cdn, 
	# 	raw_colname=raw_features_for_classify, 
	# 	z_colname=standardized_features_for_classify)
	# split_test_by_patient('../data/seed2222/z')
	# split_test_by_patient('../data/seed2222/raw')

# run()
# nmfClassfyExperiments_mean()
# if __name__ == '__main__':
# 	expri_list = []
# 	mp_list = ["0.4","0.5","0.65","0"]
# 	mc_list = ["0.1","0.3","0.4","0.6"]
# 	# os.makedirs('../data/mice/nmfresult_tmp_11-16')
# 	for minp in mp_list:
# 		for minc in mc_list:
# 			# expri_list.append([minp,minc])
# 			get_nmf_res([minp,minc])
	
	# directClassfyExperiments(expri_list[0])
	# nmfClassfyExperiments(expri_list[0])
	# pool = multiprocessing.Pool(16)
	# pool.map(nmfClassfyExperiments,expri_list)

# mp_list = ["0.4","0.5","0.65","0"]
# mc_list = ["0.1","0.3","0.4","0.6"]
# for mp in mp_list:
# 	for mc in mc_list:
# 		cdn_diff_imp = '../data/mice/mp%s_mc%s'%(str(mp),str(mc))
# 		print cdn_diff_imp
# 		if not os.path.exists(cdn_diff_imp):
# 			os.makedirs(cdn_diff_imp)
# 		for isg in [0,3]:
# 			if not os.path.exists('%s/isg%d'%(cdn_diff_imp,isg)):
# 				os.makedirs('%s/isg%d'%(cdn_diff_imp,isg))
# 				os.makedirs('%s/isg%d/pt_sg_w'%(cdn_diff_imp,isg))
# 				os.makedirs('%s/isg%d/res'%(cdn_diff_imp,isg))
# 		run(mp,mc)
# 		exit(0)

# for i in range(5):
# 	get_lastMeasures('%s/alldata_readmit_mean_test_fold%d'%(cdn_diff_imp,i),b3_cols)
# 	get_lastMeasures('%s/alldata_readmit_mean_train_fold%d'%(cdn_diff_imp,i),b3_cols)

# directClassfyExperiments()
# nmfClassfyExperiments()
# for n in [20,40,45,50,55]:
# 	rfe(n)

# clean_data('../data/alldata.csv', '../data/correction/alldata_readmit.csv')
# get_data_in_last12h('../data/correction/alldata_readmit.csv', '../data/correction/alldata_readmit_last12h.csv')
# split_nfolds('../data/correction/alldata_readmit.csv', '../data/correction/raw_shuffled/alldata_readmit', shuffle=True)

# for i in range(5):
# 	get_freq_to_trainFreq_map(i)

# def split_tr_te_by_ptids(pt, fntr, fnte):#, fnhpttid):
#     # fn = open(fnhpttid,'r')
#     i = 0
#     hpttid = {}
#     for p in pt:
#         hpttid[p] = i
#         i += 1
#     # for ln in fn:
#     #     hpttid[int(ln)] = i
#     #     i += 1
#     # fn.close()

#     # print hpttid

#     train = pd.read_csv(fntr)
#     test = pd.read_csv(fnte)

#     # np.unique(train['sid'].tolist())

#     filtered_tridlist = filter(lambda x: str(int(x)) in hpttid, np.unique(train['sid'].tolist()))
#     filtered_teidlist = filter(lambda x: str(int(x)) in hpttid, np.unique(test['sid'].tolist()))

#     # print filtered_teidlist

#     tridlist = map(lambda x: hpttid[str(int(x))], filtered_tridlist)
#     teidlist = map(lambda x: hpttid[str(int(x))], filtered_teidlist)

#     for i in range(len(filtered_tridlist)):
#     	if int(pt[tridlist[i]]) != int(filtered_tridlist[i]):
#     		print i
#     		print int(pt[tridlist[i]]),int(filtered_tridlist[i])
#     		exit('wrong')

#     for i in range(len(filtered_teidlist)):
#     	if int(pt[teidlist[i]]) != int(filtered_teidlist[i]):
#     		print i
#     		print int(pt[teidlist[i]]),int(filtered_teidlist[i])
#     		exit('wrong')
    # fn = open('../data/testsubgraph/split_index','w')
    # for i in tridlist:
    # 	fn.write(str(i)+'\n')
    # for i in teidlist:
    # 	fn.write(str(i)+'\n')
    # fn.close()

# ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
# for isg in [0,3]:
# 	for i in range(5):
# 		pt = pickle.load(open('%s/isg%d/pt_sg_w/pt_%s_fold%d'%(cdn_diff_imp,isg,'011',i),'rb'))
# 		# gt = pickle.load(open('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(cdn_diff_imp,isg,s,i),'rb'))
# 		split_tr_te_by_ptids(pt,"%s/alldata_readmit_train_fold%d.csv"%(cdn_diff_imp,i), "%s/alldata_readmit_test_fold%d.csv"%(cdn_diff_imp,i))

# for isg in [0,3]:
# 	for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
# 		for ntestth in [2,5]:
# 			mean_auc(isg, freq_t, ntestth, nc=10, nmf=False)
		# mean_auc(isg, freq_t, ntestth, nc=10, nmf=True)
# for isg in [0,3]:
# 	for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
# 		# for ntestth in [2,5]:
# 		ntestth = 2
# 		for i in range(5):
# 			# for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50, 100]:
# 			c = 1
# 			nmfclassify(isg, freq_t, ntestth, i, c)

# for ntestth in [2,5]:
# 	classify(0,'001',ntestth,0)
	# print_res_auc(0,'001',ntestth,0)
# for isg in [0,3]:
# 	for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
# 		for ntestth in [2,5]:
# 		# ntestth = 2
# 			mean_auc(isg, freq_t, ntestth)

# ftrnel = "%s/mimic_train_fold%d.nel"%(cdn_diff_imp,4)
# fnel = "%s/mimic_fold%d.nel"%(cdn_diff_imp,4)
# subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t='011', nthfold=4)

# for i in range(5):
# 	for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
# 		test_moss(freq_t,i)
# for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
# 	print freq_t, freq_to_trainFreq_map[freq_t]

# classify(3,'006',2,4,1)
# print_res_auc(3,'006',2,4,1)



# def subgraph_mining_old(tr_nel, tr_te_nel, freq_t, nthfold):
# 	'''
# 	'''
# 	print 'subgraph mining...'

# 	# fntr = '%s/mimic_m1_train_s0.%s_fold%d'%(cdn_diff_imp,freq_t,nthfold)
# 	# fntrte = '%s/mimic_m1_tr_te_s0.%s_fold%d'%(cdn_diff_imp,freq_t,nthfold)

# 	fntr = '%s/mimic_m1_train_fold%d'%(cdn_diff_imp,nthfold)
# 	fntrte = '%s/mimic_m1_tr_te_fold%d'%(cdn_diff_imp,nthfold)
	
# 	tr_freq_t = freq_to_trainFreq_map[freq_t]
# 	# tr_freq_t = freq_t

# 	moss(tr_nel, fntr, tr_freq_t)
# 	hout_tr = read_fnout('%s.out'%(fntr))
# 	hids_tr = read_fnids('%s.ids'%(fntr))

# 	moss(tr_te_nel, fntrte, freq_t)
# 	hout_tr_te = read_fnout('%s.out'%(fntrte))
# 	hids_tr_te = read_fnids('%s.ids'%(fntrte))

# 	remove_list = []
# 	add_map = {}

# 	for k in hids_tr:
# 		if k in hids_tr_te:
# 			continue
# 		for newk in hids_tr_te:
# 			if k in newk:
# 				add_map[newk] = hids_tr_te[newk]
# 				add_map[newk]['gid'] = hids_tr[k]['gid']
# 				nodeinfo = re.sub('g (\d+)','g %d'%hids_tr[k]['gid'],hids_tr_te[newk]['nodeinfo'])
# 				add_map[newk]['nodeinfo'] = nodeinfo
# 				remove_list.append(k)
# 				break

# 	for k in remove_list:
# 		hids_tr.pop(k,None)

# 	for k in add_map:
# 		hids_tr[k] = add_map[k]

# 	write_subgraphs(hids_tr, '%s/mimic_m1_fold%d.out'%(cdn_diff_imp,nthfold), '%s/mimic_m1_fold%d.ids'%(cdn_diff_imp,nthfold))
# def read_fnout_old(fnout):
# 	'''
# 	read .out file to a hashmap

# 	e.g. .out
# 	n 1 mSBP_n_-2
# 	n 2 mSBP_n_0
# 	n 3 mSBP_n_0
# 	e 2 1 tdown
# 	e 1 3 tup
# 	g 5
# 	s 3 2 5 0.0073333136 0 0.0

# 	hout[5] = 'n 1 mSBP_n_-2\nn 2 mSBP_n_0\nn 3 mSBP_n_0\ne 2 1 tdown\ne 1 3 tup\ng 5\ns 3 2 5 0.0073333136 0 0.0\n\n'
# 	'''
# 	fn = open(fnout,'r')
# 	hout = {}; nodestrlist = []; edgestrlist = []; sstr = ''; gstr = ''; gid = 0
# 	for ln in fn:
# 		ln = ln.rstrip(" \n")
# 		if len(ln) == 0:
# 			continue
# 		if ln[0] == 'n':
# 			nodestrlist.append(ln)
# 		elif ln[0] == 'e':
# 			edgestrlist.append(ln)
# 		elif ln[0] == 'g':
# 			gid = int(ln[2:])
# 			gstr = ln
# 		elif ln[0] == 's':
# 			sstr = ln
# 			nodeinfo = ''
# 			for ns in nodestrlist:
# 				nodeinfo += ns + '\n'
# 			for es in edgestrlist:
# 				nodeinfo += es + '\n'
# 			nodeinfo += gstr + '\n'
# 			nodeinfo += sstr + '\n\n'
# 			hout[gid] = nodeinfo
# 			nodestrlist = []; edgestrlist = []
# 	return hout

# def read_fnids_old(fnids, hout):
# 	'''
# 	read .ids file to a hashmap hids, then hout into hids

# 	e.g. .ids
# 	1:5242_45,6675_45,12026_45,25775_45
# 	5242_45 -> 2 
# 	6675_45 -> 4 
# 	6675_45 -> 6 
# 	12026_45 -> 4 
# 	12026_45 -> 5 
# 	12026_45 -> 6 
# 	25775_45 -> 2 
# 	25775_45 -> 5 

# 	hids['5242_45,6675_45,12026_45,25775_45'] = {gid, sgstr, nodeinfo}
# 	gid = 1
# 	sgstr = '5242_45 -> 2 \n... \n25775_45 -> 5 \n'
# 	nodeinfo = hout[gid]
# 	'''
# 	fn = open(fnids,'r')
# 	hids = {}; gid = None; prev_gid = None; sgstr = ''; nodemap = ''; sglist = ''; prev_sglist = ''
# 	i = 0
# 	for line in fn:
# 		i += 1
# 		m = re.search(r'^(\d+):(.*)$', line)
# 		if m:
# 			prev_gid = gid
# 			prev_sglist = sglist
# 			gid = int(m.group(1))
# 			sglist = m.group(2)
# 			if prev_gid != None:
# 				if prev_sglist not in hids:
# 					hids[prev_sglist] = {}
# 				hids[prev_sglist]['gid'] = prev_gid
# 				hids[prev_sglist]['sgstr'] = sgstr
# 				hids[prev_sglist]['nodeinfo'] = hout[prev_gid]
# 				nodemap = ''; sgstr = ''
# 			continue
# 		m = re.search(r'^([\d_]+) -> (.*) $', line)
# 		if m:
# 			nodemap = m.group(1) + ' -> ' + m.group(2)
# 			sgstr += nodemap + ' \n'

# 	if sglist not in hids:
# 		hids[sglist] = {}
# 	hids[sglist]['gid'] = gid
# 	hids[sglist]['sgstr'] = sgstr
# 	hids[sglist]['nodeinfo'] = hout[gid]

# 	return hids

# def write_subgraphs_old(h, fnout, fnids):
# 	'''
# 	fnout = '../data/mimic_m1_s0.%s_fold%d.out'%(freq_t,nthfold)
# 	fnids = '../data/mimic_m1_s0.%s_fold%d.ids'%(freq_t,nthfold)
# 	'''
# 	fout = open(fnout,'w')
# 	fids = open(fnids,'w')

# 	fids.write('id:list\n')
# 	ks = sorted(h.keys(), key=lambda k: h[k]['gid'])
# 	for k in ks:
# 		fids.write(str(h[k]['gid'])+':')
# 		fids.write('%s\n'%k)
# 		fids.write(h[k]['sgstr'])
# 		fout.write(h[k]['nodeinfo'])
# 	fids.close()
# 	fout.close()

