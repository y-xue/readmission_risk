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

import mimic_tensor as mt
import mimic_data as md
import mimic_classifying as mc
import nel_graph_gen_interpolation as nggi
import coding_util as cu
import preprocessing as pp
import run as rn

from rfe import *
from lr import *

import multiprocessing

def run_best_model(cdn):
	ft = 'raw'
	seed = 2222
	standardize_method = 'z'
	is_cz = False

	# cu.checkAndCreate('%s/seed%d'%(cdn,seed))
	# pp.split_nfolds('%s/alldata_readmit.csv'%cdn, '%s/seed%d/alldata_readmit'%(cdn,seed), shuffle=True, seed=seed)
	# pp.split_by_feature_type(cdn='%s/seed%d'%(cdn,seed), fn_prefix='%s/seed%d/alldata_readmit'%(cdn,seed))
	
	# cu.checkAndCreate('%s/seed%d/raw/interp'%(cdn,seed))
	# cu.checkAndCreate('%s/seed%d/raw/interp/mean/dataset'%cdn)
	# for i in range(5):
	# 	pp.impute_by_interpolation_on_last12h(
	# 		'%s/seed%d/raw/test_fold%d.csv'%(cdn,seed,i), 
	# 		'%s/seed%d/raw/interp/test_fold%d.csv'%(cdn,seed,i), 
	# 		'%s/seed%d/raw/interp/extrapolation_log_test_fold%d.txt'%(cdn,seed,i))
	# 	pp.impute_by_interpolation_on_last12h(
	# 		'%s/seed%d/raw/train_fold%d.csv'%(cdn,seed,i), 
	# 		'%s/seed%d/raw/interp/train_fold%d.csv'%(cdn,seed,i), 
	# 		'%s/seed%d/raw/interp/extrapolation_log_train_fold%d.txt'%(cdn,seed,i))
	# 	pp.impute_by_mean(
	# 		'%s/seed%d/raw/interp/train_fold%d.csv'%(cdn,seed,i),
	# 		'%s/seed%d/raw/interp/test_fold%d.csv'%(cdn,seed,i),
	# 		'%s/seed%d/raw/interp/mean/dataset/train_fold%d.csv'%(cdn,seed,i),
	# 		'%s/seed%d/raw/interp/mean/dataset/test_fold%d.csv'%(cdn,seed,i))
	# 	pp.standardize_data(
	# 		'%s/seed%d/raw/interp/mean/dataset/train_fold%d.csv'%(cdn,seed,i),
	# 		'%s/seed%d/raw/interp/mean/dataset/test_fold%d.csv'%(cdn,seed,i),
	# 		'%s/seed%d/raw/interp/mean/dataset/train_fold%d_%s.csv'%(cdn,seed,i,standardize_method),
	# 		'%s/seed%d/raw/interp/mean/dataset/test_fold%d_%s.csv'%(cdn,seed,i,standardize_method))
	
	# run temporal model
	freq_list = ['011']
	freq_to_trainFreq_map = {'011':'014'}
	nel_graph_length = 13
	
	cu.checkAndCreate('%s/seed%d/%s/interp/mean/%s'%(cdn,seed,ft,standardize_method))
	e = rn.Experiment('%s/seed%d/%s/interp/mean/%s'%(cdn,seed,ft,standardize_method),
		'%s/seed%d/%s/interp/mean/dataset'%(cdn,seed,ft),
		seed,is_cz,standardize_method,freq_list,freq_to_trainFreq_map,nel_graph_length)

	isg = 0
	freq_t = '011'
	nc = 110
	c = 2
	pl = 'l1'
	cw = 'balanced'
	ntestth = 2

	# cu.checkAndCreate('%s/isg%d'%(e.cdn,isg))
	# cu.checkAndCreate('%s/isg%d/pt_sg_w'%(e.cdn,isg))
	# cu.checkAndCreate('%s/isg%d/res'%(e.cdn,isg))
	# cu.checkAndCreate('%s/isg%d/nmf_piks'%(e.cdn,isg))

	# for foldi in range(5):
		
	# 	train = e.ftrain%(e.dataset_folder,foldi,e.standardize_method)
	# 	test = e.ftest%(e.dataset_folder,foldi,e.standardize_method)

	# 	print train
	# 	print test

	# 	ftrnel = "%s/mimic_train_fold%d.nel"%(e.cdn,foldi)
	# 	ftrnode = "%s/mimic_train_fold%d.node"%(e.cdn,foldi)
	# 	fnel = "%s/mimic_fold%d.nel"%(e.cdn,foldi)
	# 	fnode = "%s/mimic_fold%d.node"%(e.cdn,foldi)
		
	# 	e.interpolation(trcsv=train, tecsv=test, ftrnel=ftrnel, ftrnode=ftrnode, fnel=fnel, fnode=fnode)
		
	# 	e.get_freq_to_trainFreq_map(foldi)
		
	# 	for freq_t in e.moss_freq_threshold_list:
	# 		e.subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t=freq_t, foldi=foldi)

	# 		e.gen_pt_sg_files(isg, freq_t, foldi)

	# # run baseline model
	for i in range(5):
		pp.get_last_measurements(
            '%s/seed%d/raw/interp/mean/dataset/train_fold%d_%s.csv'%(cdn,seed,i,standardize_method),
            '%s/seed%d/raw/interp/mean/last_measures/dataset/train_fold%d_%s.csv'%(cdn,seed,i,standardize_method))
      	pp.get_last_measurements(
            '%s/seed%d/raw/interp/mean/dataset/test_fold%d_%s.csv'%(cdn,seed,i,standardize_method),
            '%s/seed%d/raw/interp/mean/last_measures/dataset/test_fold%d_%s.csv'%(cdn,seed,i,standardize_method))
    
	best_features = rfe('%s/seed%d/raw/interp/mean/last_measures/'%(cdn,seed), 50, standardize_method, 5, 'l1', 'balanced')
	print best_features

	best_features = ['urineByHrByWeight', 'HCT', 'INR', 'Platelets', 'RBC', 
	'DeliveredTidalVolume', 'PlateauPres', 'RAW', 'RSBI', 'mDBP', 'CV_HR', 
	'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_pH', 'Cl', 'Mg', 'Anticoagulant', 
	'beta.Blocking_agent', 'Somatostatin_preparation', 'Vasodilating_agent', 
	'AIDS', 'MetCarcinoma']

	baseline_auc = lr('%s/seed%d/raw/interp/mean/last_measures/'%(cdn,seed), standardize_method, 5, 'l1', 'balanced', 50)
	print 'baseline AUC: %s'%baseline_auc

	res_list = []
	# res_baseline_list = []
	for foldi in range(5):
		fnaddtr = '../data/seed2222/raw/interp/mean/last_measures/dataset/train_fold%d_%s.csv'%(foldi,standardize_method)
		fnaddte = '../data/seed2222/raw/interp/mean/last_measures/dataset/test_fold%d_%s.csv'%(foldi,standardize_method)
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
	# for i in range(len(res_list)):
	# i = 4
	# with open('%s/gt_te_fold%d_t'%(output_folder,i),'wb') as f:
	# 	pickle.dump(gt_te,f)
	# with open('%s/pt_te_fold%d_t'%(output_folder,i),'wb') as f:
	# 	pickle.dump(pt_te,f)
	# with open('%s/pre_te_fold%d_t'%(output_folder,i),'wb') as f:
	# 	pickle.dump(res_list[i]['n_pre_te'],f)
		# with open('%s/c_pre_te_fold%d'%(output_folder,i),'wb') as f:
		# 	pickle.dump(res_list[i]['c_pre_te'],f)

	# print res['auc_te']

# def run(self, isglist):
# 		# self.cdn = '../data/mean_last12h'
# 		# self.cdn = '../data/seed2222/%s/mice/mp%s_mc%s'%(feature_type,minp,minc)
# 		# print self.cdn
# 		cu.checkAndCreate(self.cdn)
# 		for isg in isglist:
# 			cu.checkAndCreate('%s/isg%d'%(self.cdn,isg))
# 			cu.checkAndCreate('%s/isg%d/pt_sg_w'%(self.cdn,isg))
# 			cu.checkAndCreate('%s/isg%d/res'%(self.cdn,isg))

# 		for foldi in range(5):
			
# 			train = self.ftrain%(self.dataset_folder,foldi,self.standardize_method)
# 			test = self.ftest%(self.dataset_folder,foldi,self.standardize_method)

# 			print train
# 			print test

# 			ftrnel = "%s/mimic_train_fold%d.nel"%(self.cdn,foldi)
# 			ftrnode = "%s/mimic_train_fold%d.node"%(self.cdn,foldi)
# 			fnel = "%s/mimic_fold%d.nel"%(self.cdn,foldi)
# 			fnode = "%s/mimic_fold%d.node"%(self.cdn,foldi)
			
# 			self.interpolation(trcsv=train, tecsv=test, ftrnel=ftrnel, ftrnode=ftrnode, fnel=fnel, fnode=fnode)
			
# 			self.get_freq_to_trainFreq_map(foldi)
			
# 			for freq_t in self.moss_freq_threshold_list:
# 				self.subgraph_mining(tr_nel=ftrnel, tr_te_nel=fnel, freq_t=freq_t, foldi=foldi)

# 				for isg in isglist:
# 					self.gen_pt_sg_files(isg, freq_t, foldi)

# 		# self.tuneSGParamForClassification(nmf=False)
# 		self.tuneSGParamForClassification(nmf=True)

run_best_model('../data')