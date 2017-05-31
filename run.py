import pandas as pd
import numpy as np
import subprocess
import re
import pickle
import os
import math
import time

from scipy import interp
from sklearn import metrics
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import mimic_tensor as mt
import mimic_data as md
import mimic_classifying as mc
import nel_graph_gen_interpolation as nggi
import coding_util as cu
import preprocessing as pp

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

# # without med_feature
# freq_list_no_med = ['001','002','004','006','007','009','010', '012','014','015']

class Experiment():

	def __init__(self, cdn, dataset_folder, seed, is_cz, standardize_method, freq_t_list, freq_to_trFreq, nel_graph_length):
		'''
		cdn = '../data/seed%s/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method)
		'''
		self.seed = seed
		self.cdn = cdn
		self.dataset_folder = dataset_folder
		self.is_cz = is_cz
		self.standardize_method = standardize_method
		self.ftrain = '%s/train_fold%s_%s.csv'
		self.ftest = '%s/test_fold%s_%s.csv'
		self.moss_freq_threshold_list = freq_t_list
		self.freq_to_trFreq = freq_to_trFreq
		self.nel_graph_length = nel_graph_length

		self.best_auc = 0.
		self.best_params = ''
		self.highest_tr_auc = 0.

	def interpolation(self, trcsv, tecsv, ftrnel, ftrnode, fnel, fnode):
		'''
		Do interpolation on training set (trcsv) and train+test set (trcsv+tecsv).
		Outputs: ftrnel, ftrnode, fnel, fnode.
		'''
		print 'interpolating...'
		trtecsv = '%s/trtecsv.csv'%self.cdn

		x = nggi.NelGraphGenInterpolation()
		x.scan_csv_interpolation(trcsv, ftrnel, ftrnode, cz=self.is_cz)

		tr = pd.read_csv(trcsv)
		te = pd.read_csv(tecsv)
		tr = tr.append(te)
		tr.to_csv(trtecsv, index=False)
		x.scan_csv_interpolation(trtecsv, fnel, fnode, cz=self.is_cz)

	def subgraph_mining(self, tr_nel, tr_te_nel, freq_t, foldi, cfolder=None):
		'''
		Inputs: .nel file for training set (tr_nel) and training+testing set (tr_te_nel),
				MoSS param (freq_t), fold number.
		Outpus: .out and .ids files from MoSS

		Find frequent subgraphs (A) in training set first, then find frequent subgraphs
		from training+tesing set but only select those exists in A.
		'''
		print 'subgraph mining...'

		if cfolder is None:
			fntr = '%s/mimic_m1_train_fold%d'%(self.cdn,foldi)
			fntrte = '%s/mimic_m1_tr_te_fold%d'%(self.cdn,foldi)
		else:
			fntr = '%s/mimic_m1_train_fold%d'%(cfolder,foldi)
			fntrte = '%s/mimic_m1_tr_te_fold%d'%(cfolder,foldi)
		
		print tr_nel
		print tr_te_nel
		print fntr
		print fntrte

		tr_freq_t = self.freq_to_trFreq[freq_t]

		start = time.time()

		self.moss(tr_nel, fntr, tr_freq_t)
		self.moss(tr_te_nel, fntrte, freq_t)

		print time.time() - start

		hout_tr = self.read_fnout('%s.out'%(fntr))
		hids_tr = self.read_fnids('%s.ids'%(fntr))
		hout_tr_te = self.read_fnout('%s.out'%(fntrte))
		hids_tr_te = self.read_fnids('%s.ids'%(fntrte))

		hout = {}
		hids = {}
		# (hout, hids) = self.combine_subgraph_mining_files(hout_tr, hout_tr_te, hids_tr, hids_tr_te)

		for k in hout_tr_te:
			if k in hout_tr:
				# hout[gid] = hout_tr_te[trte_gid].replace('g %d'%trte_gid,'g %d'%gid)
				trte_gid = hout_tr_te[k]['gid']
				tr_gid = hout_tr[k]['gid']
				hout[hout_tr[k]['gid']] = hout_tr_te[k]['out_str'].replace('g %d'%trte_gid,'g %d'%tr_gid)
				hids[hout_tr[k]['gid']] = hids_tr_te[hout_tr_te[k]['gid']]

		if cfolder is None:
			self.write_subgraphs(hout, hids, '%s/mimic_m1_%s_fold%d.out'%(self.cdn,freq_t,foldi), '%s/mimic_m1_%s_fold%d.ids'%(self.cdn,freq_t,foldi))
		else:
			self.write_subgraphs(hout, hids, '%s/mimic_m1_fold%d.out'%(cfolder,foldi), '%s/mimic_m1_fold%d.ids'%(cfolder,foldi))
	
	def gen_pt_sg_files(self, isg, freq_t, foldi, readonly=False):
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
		fn_sgtid = "%s/isg%d/pt_sg_w/mimic.sgtid_%s_fold%d"%(self.cdn,isg,freq_t,foldi)
		fn_ntid = "%s/isg%d/pt_sg_w/mimic.ntid_%s_fold%d"%(self.cdn,isg,freq_t,foldi)

		if not readonly:
			mt.pt_sg_w_tensor_gen("%s/mimic_fold%d.nel"%(self.cdn,foldi), 
				"%s/mimic_fold%d.node"%(self.cdn,foldi), 
				"%s/mimic_m1_%s_fold%d.out"%(self.cdn,freq_t,foldi), 
				"%s/mimic_m1_%s_fold%d.ids"%(self.cdn,freq_t,foldi), 
				fn_sparse_tensor, 
				fn_pt_sg_mat, 
				fn_pt_wd_mat, 
				fn_pt_lab, 
				fn_sgtid, 
				fn_sgs, 
				fn_ntid,
				isg=isg)

		ptsg = md.read_pt_sg_mat(fn_pt_sg_mat)
		ptwd = md.read_pt_wd_mat(fn_pt_wd_mat)
		sgs = md.read_sgs(fn_sgs)
		spt = md.read_sparse_tensor(fn_sparse_tensor)
		(hiso, hsgstr, hsgc, hsgsize) = md.sg_subiso(sgs)
		(ptsg, sgs, sptsel) = md.filter_sg(ptsg, hiso, sgs, spt=spt)
		# (ptsg, sgs) = md.filter_sg(ptsg, hiso, sgs)
		gt = md.read_pt_gt(fn_pt_gt)
		pt = md.read_pt_lab(fn_pt_lab)
		self.write_pt_sg_before_filter_pt(ptsg, ptwd, sgs, pt, gt, isg, freq_t, foldi)

	def dirClassify(self, ptsg, ptwd, sgs, pt, gt, ntestth, foldi, c, pl, cw):
		'''
		Direct classify using pt_sg.
		'''
		print 'classify:'

		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)
		(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
			self.ftrain%(self.dataset_folder,foldi,self.standardize_method),
			self.ftest%(self.dataset_folder,foldi,self.standardize_method))

		clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
		res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
		
		return res

	def nmfClassify(self, ptsg, ptwd, sgs, pt, gt, fnpik, ntestth, foldi, nc, c, pl, cw):
		'''
		Classify using NMF.
		'''
		print 'nmfclassify:'
		
		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)

		(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
			self.ftrain%(self.dataset_folder,foldi,self.standardize_method),
			self.ftest%(self.dataset_folder,foldi,self.standardize_method))
		
		clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
		# clf = SVC(C=c, kernel=pl, class_weight=cw, probability=True)
		
		# (m, clf, res) = mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
		# 	fnfmt=fnpik, nc=nc, clf=clf, fngrp='%s/isg0/011_fold%d_%d.grp'%(self.cdn,foldi,nc), sgs=sgs)
		
		(m, clf, res) = mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
			fnfmt=fnpik, nc=nc, clf=clf)

		return res

	def nmfClassify_ob(self, ptsg, ptwd, sgs, pt, gt, fnpik, ntestth, foldi, nc, c, pl, cw, fnaddtr, fnaddte, best_features):
		'''
		Classify using NMF.
		'''
		print 'nmfclassify:'
		
		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)

		(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
			self.ftrain%(self.dataset_folder,foldi,self.standardize_method),
			self.ftest%(self.dataset_folder,foldi,self.standardize_method))
		
		clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
		
		# print ptsg_tr.shape
		# (m, clf, res) = mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
		# 	fnfmt=fnpik, nc=nc, clf=clf)
		(m, clf, res, res_baseline) = mc.nmfClassify_addMoreFeature(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
			fnpik=fnpik, nc=nc, clf=clf, fnaddtr=fnaddtr, fnaddte=fnaddte, selected_features=best_features, foldi=foldi)

		return (res, gt_te, pt_te, res_baseline)

	def dirClassify_ob(self, ptsg, ptwd, sgs, pt, gt, ntestth, foldi, c, pl, cw, fnaddtr, fnaddte, best_features):
		'''
		Direct classify using pt_sg.
		'''
		print 'dirclassify_ob:'

		(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)
		(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, 
			self.ftrain%(self.dataset_folder,foldi,self.standardize_method),
			self.ftest%(self.dataset_folder,foldi,self.standardize_method))

		clf = LogisticRegression(penalty=pl, C=c, class_weight=cw)
		res = mc.dirClassify_addMoreFeature(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te,
			clf=clf, fnaddtr=fnaddtr, fnaddte=fnaddte, selected_features=best_features)

		return res

	def get_mean_auc(self, res_list):
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

		return (mean_auc, mean_auc_tr)

	def tuneCLFParamForClassification(self, freq_t, isg, ntestth, prediction_matrics, nmf=False, nc=10):
		bauc = 0.
		htauc = 0.
		bparams = ''
		fparams = 'isg%s,s%s,nc%s,c%s,%s,%s'
		detail_res = ''

		# for c in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 50]:#, 100]:
		for c in [1, 2, 5]:#, 10, 15, 20, 50]:
		# for c in [2]:
			# for pl in ['l1','l2']:
			for pl in ['l1']:
			# for pl in ['linear']:
				# for cw in ['balanced', None]:
				for cw in ['balanced']:
					res_list = []
					for foldi in range(5):
						print 'c%s,%s,%s,fold%d'%(str(c),pl,cw,foldi)
						if nmf:
							res = self.nmfClassify(prediction_matrics['ptsg'][foldi],
								prediction_matrics['ptwd'][foldi],
								prediction_matrics['sgs'][foldi],
								prediction_matrics['pt'][foldi],
								prediction_matrics['gt'][foldi],
								'%s/isg%d/nmf_piks/nmf_%s_fold%d_%d.pik'%(self.cdn,isg,freq_t,foldi,nc),
								ntestth, foldi, nc, c, pl, cw)
							with open('%s/isg%d/res/nmfres_%s_nc%d_c%s_%s_%s_fold%d'%(self.cdn,isg,freq_t,nc,str(c),pl,cw,foldi),'wb') as f:
								pickle.dump(res,f)
							# with open('%s/isg%d/svmres/nmfres_%s_nc%d_c%s_%s_%s_fold%d'%(self.cdn,isg,freq_t,nc,str(c),pl,cw,foldi),'wb') as f:
							# 	pickle.dump(res,f)
						else:
							res = self.dirClassify(prediction_matrics['ptsg'][foldi],
								prediction_matrics['ptwd'][foldi],
								prediction_matrics['sgs'][foldi],
								prediction_matrics['pt'][foldi],
								prediction_matrics['gt'][foldi],
								ntestth, foldi, c, pl, cw)
							with open('%s/isg%d/res/dirres_%s_c%s_%s_%s_fold%d'%(self.cdn,isg,freq_t,str(c),pl,cw,foldi),'wb') as f:
								pickle.dump(res,f)
							# with open('%s/isg%d/svmres/dirres_%s_c%s_%s_%s_fold%d'%(self.cdn,isg,freq_t,str(c),pl,cw,foldi),'wb') as f:
							# 	pickle.dump(res,f)
						res_list.append(res)
					(auc, tr_auc) = self.get_mean_auc(res_list)

					detail_res = detail_res + fparams%(isg,freq_t,nc,c,pl,cw) + ': %f\n'%auc
					print detail_res

					if auc > bauc:
						bauc = auc
						bparams = fparams%(isg,freq_t,nc,c,pl,cw)
					htauc = max(htauc,tr_auc)
					print 'current nc bauc: %f, params: %s'%(bauc,bparams)
					print 'Best auc: %f, params: %s'%(self.best_auc,self.best_params)

		return (bauc, bparams, htauc, detail_res)

	def write_pt_sg_before_filter_pt(self, ptsg, ptwd, sgs, pt, gt, isg, s, i):
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

	def read_prediction_matrics(self, isg, freq_t, cfolder=None):
		if cfolder is None:
			cfolder = '%s/isg%d/pt_sg_w'%(self.cdn,isg)
		prediction_matrics = {}
		prediction_matrics['ptsg'] = []
		prediction_matrics['ptwd'] = []
		prediction_matrics['sgs'] = []
		prediction_matrics['pt'] = []
		prediction_matrics['gt'] = []
		for foldi in range(5):
			ptsg = pickle.load(open('%s/ptsg_%s_fold%d'%(cfolder,freq_t,foldi),'rb'))
			ptwd = pickle.load(open('%s/ptwd_%s_fold%d'%(cfolder,freq_t,foldi),'rb'))
			sgs = pickle.load(open('%s/sgs_%s_fold%d'%(cfolder,freq_t,foldi),'rb'))
			pt = pickle.load(open('%s/pt_%s_fold%d'%(cfolder,freq_t,foldi),'rb'))
			gt = pickle.load(open('%s/gt_%s_fold%d'%(cfolder,freq_t,foldi),'rb'))

			prediction_matrics['ptsg'].append(ptsg)
			prediction_matrics['ptwd'].append(ptwd)
			prediction_matrics['sgs'].append(sgs)
			prediction_matrics['pt'].append(pt)
			prediction_matrics['gt'].append(gt)

		return prediction_matrics

	def moss(self, fin, fout_prefix, freq_t):
		'''
		Call Java package MoSS.
		e.g. java -cp ../moss.jar moss.Miner -inelist -onelist -m1 -s0.011 ../mimic_fold2.nel ../observer/train_fold2.out ../observer/train_fold2.ids
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
		print ' '.join(cmd)
		with open('%s_%s_moss_log'%(fout_prefix,freq_t), 'w') as f:
			subprocess.call(cmd, stderr=f)

	def read_fnout(self, fnout):
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

		hout['mSBP -2 0 0'] = {
		'gid': 5,
		'out_str': n 1 mSBP_n_-2\nn 2 mSBP_n_0\nn 3 mSBP_n_0\ne 2 1 tdown\ne 1 3 tup\ng 5\ns 3 2 5 0.0073333136 0 0.0\n\n'
		}
		'''
		print 'reading fnout..'
		hout = {}
		hn = {}; he = {}
		out_str = ''
		fn = open(fnout,'r')
		for ln in fn:
			ln = ln.rstrip(" \n")
			# mn = re.search(r'^n (\d+) (\S+_n|\S+_m|\S+_p|loc)_(\S+)$', ln)
			out_str += ln + '\n'
			mn = re.search(r'^n (\d+) (\S+_n|\S+_r|\S+_m|\S+_p|loc)_(\S+)$', ln)
			if mn:
				nid = int(mn.group(1))
				test = mn.group(2)
				val = mn.group(3)
				hn[nid] = val # hn:{1:-1,2:0,3:0,4:0,5:0,6:0}

			me = re.search(r'^e (\d+) (\d+) (\S+)', ln)
			if me:
				nfrom = int(me.group(1))
				nto = int(me.group(2))
				he[nfrom] = nto

			mg = re.search(r'^g (\d+)', ln)
			if mg:
				gid = int(mg.group(1))

			ms = re.search(r'^s (\d+) \d+ (\d+)', ln)
			if ms:
				# s 6 5 1 0.0011698232 0 0.0
				# nnodes = 6, sup = 1
				nnodes = int(ms.group(1))
				sup = int(ms.group(2))
				nstart = set(he.keys()).difference(set(he.values()))
				if len(nstart) == 1:
					niter = list(nstart)[0]
				elif len(nstart) == 0:
					niter = 1
				else:
					print("warning: sg %s has >1 start" % (gid))
				if niter not in hn:
					print 'niter not in hn: ', niter
				sgstr = "%s" % (hn[niter])

				# go through the map
				while he.has_key(niter):
					sgstr += " %s" % (hn[he[niter]])
					niter = he[niter]
				hn = {}; he = {}

				key = '%s\t%s'%(test,sgstr)
				# print gid
				# print test
				# print sgstr
				# print key
				# print out_str
				if key in hout:
					print key
					print hout[key]
					# print out_str
					exit('error in read fnout.')

				hout[key] = {}
				hout[key]['gid'] = gid
				hout[key]['out_str'] = out_str + '\n'

				# if gid == 1:
				# 	print out_str
				# 	exit(0)
				out_str = ''

		return hout

	def read_fnids(self, fnids):
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
		print 'reading fnids..'
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

	# def getKeyFromKeylistContainsK(self, k=None, keylist=[]):
	# 	for key in keylist:
	# 		if k in key:
	# 			return key
	# 	return None

	# def combine_subgraph_mining_files(self, hout_tr, hout_tr_te, hids_tr, hids_tr_te):
	# 	hids = {}
	# 	hout = {}
	# 	for k in hids_tr:
	# 		gid = hids_tr[k]['gid']
	# 		newkey = self.getKeyFromKeylistContainingK(k=k, keylist=hids_tr_te.keys())
	# 		if newkey != None:
	# 			if newkey in hids:
	# 				exit('error in combine_subgraph_mining_files.')
	# 			hids[newkey] = {}
	# 			hids[newkey]['gid'] = gid
	# 			hids[newkey]['sgstr'] = hids_tr_te[newkey]['sgstr']

	# 			trte_gid = hids_tr_te[newkey]['gid']
	# 			hout[gid] = hout_tr_te[trte_gid].replace('g %d'%trte_gid,'g %d'%gid)
	# 		else:
	# 			print 'k in tr not in trte: %s'%k
	# 			hids[k] = hids_tr[k]
	# 			hout[gid] = hout_tr[hid]
	# 	return (hout, hids)

	# def read_fnout_old(self, fnout):
	# 	'''
	# 	THIS IS WRONG!
	# 	'''
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

	# 	hout['n 1 mSBP_n_-2\nn 2 mSBP_n_0\nn 3 mSBP_n_0\ne 2 1 tdown\ne 1 3 tup\n'] = 
	# 		{'gid': 5, 
	# 		'nodeinfo' = n 1 mSBP_n_-2\nn 2 mSBP_n_0\nn 3 mSBP_n_0\ne 2 1 tdown\ne 1 3 tup\ng 5\ns 3 2 5 0.0073333136 0 0.0\n\n'}
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

	# 			key = ''
	# 			for ns in nodestrlist:
	# 				key += ns + '\n'
	# 			for es in edgestrlist:
	# 				key += es + '\n'

	# 			if key in hout:
	# 				exit('error in read_fnout.')
	# 			hout[key] = {}
	# 			hout[key]['gid'] = gid
	# 			hout[key]['gstr'] = gstr + '\n'
	# 			hout[key]['sstr'] = sstr + '\n\n'
	# 			nodestrlist = []; edgestrlist = []
	# 	return hout

	# def read_fnids_old(self, fnids):
	# 	'''
	# 	THIS IS WRONG!
	# 	'''
	# 	'''
	# 	read .ids file to a hashmap hids

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

	# 	hids[1] = '5242_45,6675_45,12026_45,25775_45\n5242_45 -> 2 \n...25775_45 -> 5 \n'
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
	# 				sginfo = prev_sglist + '\n' + sgstr
	# 				hids[prev_gid] = sginfo
	# 				nodemap = ''; sgstr = ''
	# 			continue
	# 		m = re.search(r'^([\d_]+) -> (.*) $', line)
	# 		if m:
	# 			nodemap = m.group(1) + ' -> ' + m.group(2)
	# 			sgstr += nodemap + ' \n'

	# 	sginfo = sglist + '\n' + sgstr
	# 	hids[gid] = sginfo

	# 	return hids

	def write_subgraphs(self, hout, hids, fnout, fnids):
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
			fout.write(hout[k].lstrip('\n'))
		fout.close()

	def get_freq_to_trainFreq_map(self, foldi):
		'''
		Given freq_t list, get corresponding freq_t for training set.
		'''
		print 'fold%d:'%foldi
		ftrnel="%s/mimic_train_fold%d.nel"%(self.cdn,foldi)
		fnel = "%s/mimic_fold%d.nel"%(self.cdn,foldi)
		with open(ftrnel) as f:
			cnt_tr_graphs = sum(1 for _ in f)/self.nel_graph_length
		with open(fnel) as f:
			cnt_tr_te_graphs = sum(1 for _ in f)/self.nel_graph_length
		
		l = []

		for freq_t in self.moss_freq_threshold_list:
			freq = math.ceil(float('0.'+freq_t)*cnt_tr_te_graphs/100)
			tr_freq_t = math.floor(math.ceil(cnt_tr_te_graphs*float('0.'+freq_t)/100)*100000/cnt_tr_graphs)*0.001
			tr_freq_t = str(tr_freq_t)[2:]
			tr_freq = math.ceil(float('0.'+tr_freq_t)*cnt_tr_graphs/100)
			l.append(float('0.'+tr_freq_t))
		
		print l

		return l

	def get_detail_nmfres(self):
		detail_res = ''
		for isg in [0]:
			for freq_t in self.moss_freq_threshold_list:
				for nc in [60,70,80,90,100,110,120]:
					for c in [1, 2, 5]:
						for pl in ['l1']:
							for cw in ['balanced']:
								res_list = []
								for foldi in range(5):
									print 'c%s,%s,%s,fold%d'%(str(c),pl,cw,foldi)
									res = pickle.load(open('%s/isg%d/res/nmfres_%s_nc%d_c%s_%s_%s_fold%d'%(self.cdn,isg,freq_t,nc,str(c),pl,cw,foldi),'r'))
									res_list.append(res)
								(auc, tr_auc) = self.get_mean_auc(res_list)

								detail_res = detail_res + 'isg%s,s%s,nc%s,c%s,%s,%s'%(isg,freq_t,nc,c,pl,cw) + ': %f\n'%auc
								print detail_res

		fndr = open('%s/detailResult_isg%d.txt'%(self.cdn,isg),'w')
		fndr.write(detail_res)
		fndr.close()
		
	def tuneSGParamForClassification(self, nmf=False):
		print self.cdn
		output = ''
		ntestth = 2
		detail_res = ''
		for isg in [0]:
			output += 'isg %d:\n'%isg
			if nmf:
				cu.checkAndCreate('%s/isg%d/nmf_piks'%(self.cdn,isg))
			
			for freq_t in self.moss_freq_threshold_list:
				output += 'freq_t %s: '%freq_t
				
				prediction_matrics = self.read_prediction_matrics(isg,freq_t)

				if nmf:
					bauc = 0.
					btauc = 0.
					bparams = ''
					# for nc in [10,20,30,40,50,60,70,80,90,100,110,120]:
					# for nc in [60,70,80,90,100,110,120]:
					if freq_t in ['012','013']:
						nc_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
					else:
						nc_list = [10,20,30,40,50,110,130,140,150]
					for nc in nc_list:
					# for nc in [90]:
						(tbauc,tbparams,tbtauc, dres) = self.tuneCLFParamForClassification(freq_t, isg, ntestth, prediction_matrics, nmf=True, nc=nc)
						if tbauc > bauc:
							bauc = tbauc
							bparams = tbparams
							btauc = tbtauc

							if bauc > self.best_auc:
								self.best_auc = bauc
								self.best_params = bparams
								self.highest_tr_auc = btauc

							tfn = open('%s/nmfCurResult_isg%s_s%s.txt'%(self.cdn,isg,freq_t),'w')
							tfn.write('%f (%s)\n'%(bauc,bparams))
							tfn.close()
						detail_res += dres
				else:
					(bauc,bparams,btauc,dres) = self.tuneCLFParamForClassification(freq_t, isg, ntestth, prediction_matrics, nmf=False)
					detail_res += dres
			
				output += '%f (%s)\n'%(bauc,bparams)
				if nmf:
					tfn = open('%s/nmfResult_isg%s_s%s.txt'%(self.cdn,isg,freq_t),'w')
					tfn.write(output)
					tfn.close()
			output += '\n'
		if nmf:
			fn = open('%s/nmfResult_isg%d.txt'%(self.cdn,isg),'w')
		else:
			fn = open('%s/dirResult_isg%d.txt'%(self.cdn,isg),'w')
		fn.write(output)

		if nmf:
			fndr = open('%s/detailResult_isg%d.txt'%(self.cdn,isg),'w')
		else:
			fndr = open('%s/detailDirResult_isg%d.txt'%(self.cdn,isg),'w')
		fndr.write(detail_res)
		fndr.close()
		fn.close()

	def run(self, isglist):
		# self.cdn = '../data/mean_last12h'
		# self.cdn = '../data/seed2222/%s/mice/mp%s_mc%s'%(feature_type,minp,minc)
		# print self.cdn
		cu.checkAndCreate(self.cdn)
		for isg in isglist:
			cu.checkAndCreate('%s/isg%d'%(self.cdn,isg))
			cu.checkAndCreate('%s/isg%d/pt_sg_w'%(self.cdn,isg))
			cu.checkAndCreate('%s/isg%d/res'%(self.cdn,isg))

		for foldi in range(5):
			
			train = self.ftrain%(self.dataset_folder,foldi,self.standardize_method)
			test = self.ftest%(self.dataset_folder,foldi,self.standardize_method)

			print train
			print test

			ftrnel = "%s/mimic_train_fold%d.nel"%(self.cdn,foldi)
			ftrnode = "%s/mimic_train_fold%d.node"%(self.cdn,foldi)
			fnel = "%s/mimic_fold%d.nel"%(self.cdn,foldi)
			fnode = "%s/mimic_fold%d.node"%(self.cdn,foldi)
			
			self.interpolation(trcsv=train, tecsv=test, ftrnel=ftrnel, ftrnode=ftrnode, fnel=fnel, fnode=fnode)
			
			self.get_freq_to_trainFreq_map(foldi)
			
			for freq_t in self.moss_freq_threshold_list:
				self.subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t=freq_t, foldi=foldi)

				for isg in isglist:
					self.gen_pt_sg_files(isg, freq_t, foldi)

		# self.tuneSGParamForClassification(nmf=False)
		self.tuneSGParamForClassification(nmf=True)

if __name__ == '__main__':
	# pp.clean_data('../data/alldata.csv', '../data/alldata_readmit.csv')
	# cdn = '../data/seed2222'
	# cu.checkAndCreate(cdn)
	# pp.split_nfolds('../alldata_readmit.csv', '../data/seed2222/alldata_readmit', shuffle=True, seed=2222)
	# pp.split_by_feature_type(fn_prefix='%s/alldata_readmit'%cdn, 
	# 	raw_colname=raw_features_for_classify, 
	# 	z_colname=standardized_features_for_classify)
	# pp.split_test_by_patient('../data/seed2222/z')
	# train = '../data/seed2222/%s/mice/train_mp%s_mc%s_fold%d.csv'%(feature_type,minp,minc,foldi)
	# test = '../data/seed2222/%s/mice/test_mp%s_mc%s_fold%d.csv'%(feature_type,minp,minc,foldi)

	# one hour interval:
	# pp.clean_data('../data/alldata.csv', '../data/alldata_readmit.csv')
	# cdn = '../data/seed2222_one_hour_interval'
	# cu.checkAndCreate(cdn)
	# pp.split_nfolds('../data/alldata_readmit_one_hour_interval.csv', '../data/seed2222_one_hour_interval/alldata_readmit', shuffle=True, seed=2222)
	# pp.split_by_feature_type(cdn=cdn,
	# 	fn_prefix='%s/alldata_readmit'%cdn, 
	# 	raw_colname=raw_features_for_classify, 
	# 	z_colname=standardized_features_for_classify)
	# pp.split_test_by_patient('../data/seed2222_one_hour_interval/raw',
	# 	'../data/seed2222_one_hour_interval/raw/mice/imputing', '')

	# for i in range(5):
	# 	pp.standardize_data(
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/train_fold%d_z.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/test_fold%d_z.csv'%i)
	# for i in range(5):
	# 	pp.cz_standardize_data(
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/train_fold%d_cz.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mice/mp0.5_mc0.6/dataset/test_fold%d_cz.csv'%i)
	# pp.split_test_by_patient('../data/seed2222_one_hour_interval/raw',
	# 	'../data/seed2222_one_hour_interval/raw/standardize_z_mice/imputing', '_z')
	# run()

	# for i in range(5):
	# 	pp.impute_by_mean(
	# 		'../data/seed2222/raw/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/test_fold%d.csv'%i,
	# 		'../data/seed2222/raw/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/mean/dataset/test_fold%d.csv'%i)
	# 	pp.standardize_data(
	# 		'../data/seed2222/raw/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/mean/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222/raw/mean/dataset/train_fold%d_z.csv'%i,
	# 		'../data/seed2222/raw/mean/dataset/test_fold%d_z.csv'%i)
		# pp.cz_standardize_data(
		# 	'../data/seed2222/raw/mean/dataset/train_fold%d.csv'%i,
		# 	'../data/seed2222/raw/mean/dataset/test_fold%d.csv'%i,
		# 	'../data/seed2222/raw/mean/dataset/train_fold%d_cz.csv'%i,
		# 	'../data/seed2222/raw/mean/dataset/test_fold%d_cz.csv'%i)

	# cu.checkAndCreate('../data/seed2222_one_hour_interval/raw/mean/dataset')
	# for i in range(5):
	# 	pp.impute_by_mean(
	# 		'../data/seed2222_one_hour_interval/raw/train_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/test_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/test_fold%d.csv'%i)
	# 	pp.standardize_data(
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/train_fold%d_z.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/test_fold%d_z.csv'%i)
	# 	pp.cz_standardize_data(
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/train_fold%d_cz.csv'%i,
	# 		'../data/seed2222_one_hour_interval/raw/mean/dataset/test_fold%d_cz.csv'%i)

	# Nov 2016
	# expri_list = []
	# mp_list = ["0.4","0.5","0.65","0"]
	# mc_list = ["0.1","0.3","0.4","0.6"]
	# for minp in mp_list:
	# 	for minc in mc_list:
	# 		expri_list.append([minp,minc])
	# pool = multiprocessing.Pool(16)
	# pool.map(nmfClassfyExperiments,expri_list)
	# ft = 'z'
	

	# for i in range(5):
	# 	pp.standardize_data(
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/train_fold%d_z.csv'%i,
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/test_fold%d_z.csv'%i)
	# for i in range(5):
	# 	pp.cz_standardize_data(
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/train_fold%d_cz.csv'%i,
	# 		'../data/seed2222/raw/interp/mice/mp0.5_mc0.6/dataset/test_fold%d_cz.csv'%i)

	# for i in range(5):
	# 	pp.impute_by_mean(
	# 		'../data/seed2222/raw/interp/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/test_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/test_fold%d.csv'%i)

	# for i in range(5):
	# 	pp.standardize_data(
	# 		'../data/seed2222/raw/interp/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/train_fold%d_z.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/test_fold%d_z.csv'%i)
	# for i in range(5):
	# 	pp.cz_standardize_data(
	# 		'../data/seed2222/raw/interp/mean/dataset/train_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/test_fold%d.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/train_fold%d_cz.csv'%i,
	# 		'../data/seed2222/raw/interp/mean/dataset/test_fold%d_cz.csv'%i)

	# interp
	ft = 'raw'
	minp = 0.5
	minc = 0.6
	seed = 2222

	# standardize_method = "cz"
	# is_cz = True
	standardize_method = "z"
	is_cz = False

	freq_list = ['001','002','003','004','005','006','008','009','01','011']#,'012','013']
	freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}#,'012':'015','013':'017'}
	# freq_list = ['005']
	# freq_to_trainFreq_map = {'005':'007'}
	nel_graph_length = 13
	cu.checkAndCreate('../data/seed%s/%s/interp/mean/%s'%(seed,ft,standardize_method))
	e = Experiment('../data/seed%s/%s/interp/mean/%s'%(seed,ft,standardize_method),
		'../data/seed%s/%s/interp/mean/dataset'%(seed,ft),
		seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)
	# cu.checkAndCreate('../data/seed%s/%s/interp/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method))
	# e = Experiment('../data/seed%s/%s/interp/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method),
	# 	'../data/seed%s/%s/interp/mice/mp%s_mc%s/dataset'%(seed,ft,minp,minc),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# e.run([0])
	# for foldi in range(5):
	# 	for freq_t in e.moss_freq_threshold_list:
	# 		ftrnel = "%s/mimic_train_fold%d.nel"%(e.cdn,foldi)
	# 		ftrnode = "%s/mimic_train_fold%d.node"%(e.cdn,foldi)
	# 		fnel = "%s/mimic_fold%d.nel"%(e.cdn,foldi)
	# 		fnode = "%s/mimic_fold%d.node"%(e.cdn,foldi)
	# 		e.subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t=freq_t, foldi=foldi)


	# cu.checkAndCreate('%s/isg%d'%(e.cdn,0))
	# cu.checkAndCreate('%s/isg%d/pt_sg_w'%(e.cdn,0))
	# cu.checkAndCreate('%s/isg%d/res'%(e.cdn,0))
	
	# for foldi in range(5):
	# 	for freq_t in e.moss_freq_threshold_list:
	# 		e.gen_pt_sg_files(1, freq_t, foldi, readonly=True)

	# for foldi in range(5):
	# 	ftrnel = "%s/mimic_train_fold%d.nel"%(e.cdn,foldi)
	# 	ftrnode = "%s/mimic_train_fold%d.node"%(e.cdn,foldi)
	# 	fnel = "%s/mimic_fold%d.nel"%(e.cdn,foldi)
	# 	fnode = "%s/mimic_fold%d.node"%(e.cdn,foldi)
	# 	for freq_t in ['012','013']:
	# 		e.subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t=freq_t, foldi=foldi)

	# 		e.gen_pt_sg_files(0, freq_t, foldi)

	e.tuneSGParamForClassification(nmf=False)
	# e.get_detail_nmfres()



	# mean
	# ft = 'raw'
	# seed = 2222


	# on server:
	# freq_t 004: 0.618911 (isg0,s004,nc70,c1,l1,balanced)
	# 
	# on mac:
	# freq_t 004: 0.607603 (isg0,s004,nc70,c1,l1,balanced)
	# freq_t 005: 0.597428 (isg0,s005,nc90,c2,l1,balanced)
	# 
	# Dec 2016
	# ft = 'raw'
	# minp = 0.5
	# minc = 0.6
	# seed = 2222

	# standardize_method = "cz"
	# is_cz = True
	# standardize_method = "z"
	# is_cz = False

	# freq_list = ['001','002','003','004','005','006','008','009', '01','011']
	# freq_to_trainFreq_map = {'001':'001','002':'002','003':'004','004':'005','005':'007','006':'008','008':'01','009':'011','01':'013','011':'014'}
	# freq_list = ['005']
	# freq_to_trainFreq_map = {'005':'007'}
	# nel_graph_length = 13

	# mean:
	# cu.checkAndCreate('../data/seed%s/%s/mean/%s'%(seed,ft,standardize_method))
	# e = Experiment('../data/seed%s/%s/mean/%s'%(seed,ft,standardize_method),
	# 	'../data/seed%s/%s/mean/dataset'%(seed,ft),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# cu.checkAndCreate('../data/seed%s_one_hour_interval/%s/mean/%s'%(seed,ft,standardize_method))
	# e = Experiment('../data/seed%s_one_hour_interval/%s/mean/%s'%(seed,ft,standardize_method),
	# 	'../data/seed%s_one_hour_interval/%s/mean/dataset'%(seed,ft),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# e.run()

	# mice:
	# cu.checkAndCreate('../data/seed%s_one_hour_interval/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method))
	# e = Experiment('../data/seed%s_one_hour_interval/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method),
	# 	'../data/seed%s_one_hour_interval/%s/mice/mp%s_mc%s/dataset'%(seed,ft,minp,minc),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# cu.checkAndCreate('../data/seed%s_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method))
	# e = Experiment('../data/seed%s_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method),
	# 	'../data/seed%s_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/dataset'%(seed,ft,minp,minc),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# cu.checkAndCreate('../data/seed%s/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method))
	# e = Experiment('../data/seed%s/%s/mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method),
	# 	'../data/seed%s/%s/mice/mp%s_mc%s/dataset'%(seed,ft,minp,minc),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# cu.checkAndCreate('../data/seed%s/%s/standardize_z_mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method))
	# e = Experiment('../data/seed%s/%s/standardize_z_mice/mp%s_mc%s/%s'%(seed,ft,minp,minc,standardize_method),
	# 	'../data/seed%s/%s/standardize_z_mice/mp%s_mc%s/dataset'%(seed,ft,minp,minc),
	# 	seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	# e.run()
	# e.tuneSGParamForClassification(nmf=False)
	# e.tuneSGParamForClassification(nmf=True)

	# print e.ftrain%(e.dataset_folder,0,e.standardize_method)
	# print e.ftest%(e.dataset_folder,0,e.standardize_method)
	# for foldi in range(5):
	# 	pp.standardize_data(e.ftrain%(e.cdn,foldi,''), e.ftest%(e.cdn,foldi,''),
	# 		e.ftrain%(e.cdn,foldi,'_z'), e.ftest%(e.cdn,foldi,'_z'))
	# 	pp.cz_standardize_data(e.ftrain%(e.cdn,foldi,''), e.ftest%(e.cdn,foldi,''),
	# 		e.ftrain%(e.cdn,foldi,'_cz'), e.ftest%(e.cdn,foldi,'_cz'))
	
	# e.tuneSGParamForClassification(nmf=False)
	# e.tuneSGParamForClassification(nmf=True)
	# while (not os.path.isfile('%s/isg%d/pt_sg_w/gt_%s_fold%d'%(e.cdn,3,'011',4))):
	# 	print 'waiting..%s/isg%d/pt_sg_w/gt_%s_fold%d'%(e.cdn,3,'011',4)
	# 	time.sleep(600)
	# e.tuneSGParamForClassification(nmf=True)
	# e.tuneSGParamForClassification(nmf=False)
	# print e.cdn
	# print e.ftrain%(e.cdn,0)
	# print e.ftest%(e.cdn,0)
	# print e.moss_freq_threshold_list

