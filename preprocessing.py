"""
Data preprocessing: splitting, normalization, imputation, breaking into time intervals.
Created by yxue - 05-18-2017
"""
import pandas as pd
import numpy as np
import coding_util as cu
# from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold

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


labs = ['Creatinine', 'BUN', 'BUNtoCr', 
'urineByHrByWeight', 'eGFR', 'AST', 'ALT', 'TBili', 'DBili', 'Albumin', 'tProtein', 
'ASTtoALT', 'HCT', 'Hgb', 'INR', 'Platelets', 'PT', 'PTT', 'RBC', 'WBC', 'RESP', 
'mSaO2', 'PaO2toFiO2', 'MinuteVent', 'DeliveredTidalVolume', 'FiO2Set', 'PEEPSet', 
'PIP', 'PlateauPres', 'RAW', 'RSBI', 'RSBIRate', 'mSBP', 'mDBP', 'mMAP', 'CV_HR', 
'mCrdIndx', 'mCVP', 'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_PaO2', 'Art_pH', 'Na', 
'K', 'Cl', 'Glucose', 'Ca', 'Mg', 'IonCa', 'Lactate', 'GCS', 'temp', 'Age']

# Kidney
Creatinine=(0.6,1.1)
BUN=(7,20)
BUNtoCr=(10,20) # have 10
urineByHrByWeight=(0.5,400) #400=30000/75kg
eGFR=(90,10000)

# liver
AST=(8,40)
ALT=(10,55)
TBili=(0.2,1.4)
DBili=(0,0.4)
Albumin=(3.5,5.5)
tProtein=(6.3,8.2)
ASTtoALT=(0,1)

# Hemat/GastroIntestinal
HCT=(37,50)
Hgb=(13.5,15.5) # c(12,18), For men, 13.5 to 17.5 grams per deciliter. For women, 12.0 to 15.5
INR=(0.8,1.099) # c(0.9,1.2)
Platelets=(140,400)
PT=(10,15)
PTT=(18,45)
RBC=(4.4, 5) # c(4,5.5)
WBC=(4.5,10.5) #c(4.5,11)

# vitals
RESP=(12,20)
mSaO2=(95,100) # c(88,100)

# Gas Exchange
PaO2toFiO2=(0,0)

# Breathing
MinuteVent=(10,15) # flow: tidalvolobs*RESP
DeliveredTidalVolume=(400,600)
FiO2Set=(0,0.5) # < 50%

# rapid shallow breathing index
PEEPSet=(5,8)
PIP=(0,35)
PlateauPres=(0,30)

# Change in Airway Resistance  
RAW=(0,5)

RSBI=(20,105)
RSBIRate=(0,20) # <20%

# Hemodynamic and circulation: blood output and O2/CO2 transport
mSBP=(90,140)
mDBP=(60,90)
mMAP=(70,105)
CV_HR= (60,100)
mCrdIndx=(2.6,4) # c(2.4,4)
mCVP=(3,8) # c(1,6)

# AcidBase
Art_BE=(-3,3) #c(-2, 2)
Art_CO2=(23,26) #c(22,26) c(23,30)
Art_PaCO2=(33,45) #c(35,45)
Art_PaO2=(75,100) #c(75,105)
Art_pH=(7.35,7.439) #c(7.35, 7.45)

# Electrolytes
Na=(135,147) #c(135, 145)
K=(3.5,5.1) #c(3.5, 5.0)
Cl=(95,104.999) #c(95, 110)
Glucose=(80,120) # c(60,120)
Ca=(8.5, 10.2) # c(8.4,10.5)
Mg=(1.5,2.0) #c(1.5,2.3) 
IonCa=(1.1,1.25) # c(1.03,1.25)
Lactate=(4.5,14)

# General & Central Nervous System
GCS=(13,15)
temp=(97,98.4)
Age=(18,40) # lowesr risk range

reference_range = {'Creatinine':Creatinine, 'BUN':BUN, 'BUNtoCr':BUNtoCr, 
'urineByHrByWeight':urineByHrByWeight, 'eGFR':eGFR, 'AST':AST, 'ALT':ALT, 
'TBili':TBili, 'DBili':DBili, 'Albumin':Albumin, 'tProtein':tProtein, 
'ASTtoALT':ASTtoALT, 'HCT':HCT, 'Hgb':Hgb, 'INR':INR, 'Platelets':Platelets, 
'PT':PT, 'PTT':PTT, 'RBC':RBC, 'WBC':WBC, 'RESP':RESP, 'mSaO2':mSaO2, 
'PaO2toFiO2':PaO2toFiO2, 'MinuteVent':MinuteVent, 
'DeliveredTidalVolume':DeliveredTidalVolume, 'FiO2Set':FiO2Set, 'PEEPSet':PEEPSet, 
'PIP':PIP, 'PlateauPres':PlateauPres, 'RAW':RAW, 'RSBI':RSBI, 'RSBIRate':RSBIRate, 
'mSBP':mSBP, 'mDBP':mDBP, 'mMAP':mMAP, 'CV_HR':CV_HR, 'mCrdIndx':mCrdIndx, 
'mCVP':mCVP, 'Art_BE':Art_BE, 'Art_CO2':Art_CO2, 'Art_PaCO2':Art_PaCO2, 
'Art_PaO2':Art_PaO2, 'Art_pH':Art_pH, 'Na':Na, 'K':K, 'Cl':Cl, 'Glucose':Glucose, 
'Ca':Ca, 'Mg':Mg, 'IonCa':IonCa, 'Lactate':Lactate, 'GCS':GCS, 'temp':temp, 'Age':Age}

toffset = 720

# print reference_range
# print reference_range['FiO2Set']

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
	data.to_csv(fout,index=False)

def split_nfolds(fin, fout_prefix, shuffle=False, seed=2222):
	print 'split_nfolds'
	df = pd.read_csv(fin)
	gp = df.groupby('sid')
	# kf = KFold(n_splits=5, shuffle=shuffle, random_state=seed)
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

def split_by_feature_type(cdn, fn_prefix):
	'''
	split data into two sets, one contains raw features + medical features, 
	another contains standardized features + medical features.
	'''
	print 'split_by_feature_type'
	for i in range(5):
		training = pd.read_csv('%s_train_fold%d.csv'%(fn_prefix,i))
		testing = pd.read_csv('%s_test_fold%d.csv'%(fn_prefix,i))
		raw_train = training[raw_features_for_classify]
		raw_test = testing[raw_features_for_classify]
		# z_train = training[standardized_features_for_classify]
		# z_test = testing[standardized_features_for_classify]
		cu.checkAndCreate('%s/raw/'%cdn)
		# cu.checkAndCreate('%s/z/'%cdn)
		raw_train.to_csv('%s/raw/train_fold%d.csv'%(cdn,i),index=False)
		raw_test.to_csv('%s/raw/test_fold%d.csv'%(cdn,i),index=False)
		# z_train.to_csv('%s/z/train_fold%d.csv'%(cdn,i),index=False)
		# z_test.to_csv('%s/z/test_fold%d.csv'%(cdn,i),index=False)

def split_test_by_patient(cdn, out_folder, suffix=''):
	'''
	This is a help function for MICE imputation.
	Given folder, extract data for each patient to be used in MICE
	'''
	for i in range(5):
		test = pd.read_csv('%s/test_fold%d%s.csv'%(cdn,i,suffix))
		gp = test.groupby('sid')
		cu.checkAndCreate('%s/test_fold%d'%(out_folder,i))
		fn = open('%s/test_fold%d/sid_list.txt'%(out_folder,i),'w')
		for sid, group in gp:
			group.to_csv('%s/test_fold%d/%d.csv'%(out_folder,i,sid),index=False)
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

def standardize_data(fntr, fnte, fntr_z, fnte_z):
	'''
	Given training/testing set, standardize them.
	'''
	print 'standardizing data...'
	tr = pd.read_csv(fntr)
	te = pd.read_csv(fnte)
	mn = tr.mean()
	sd = tr.std()
	col_list = tr.columns
	tr_z = pd.DataFrame(columns=col_list)
	te_z = pd.DataFrame(columns=col_list)
	for col in col_list:
		print col
		if col in labs:
			tr_z[col] = (tr[col] - mn[col]) / sd[col]
			te_z[col] = (te[col] - mn[col]) / sd[col]
		else:
			tr_z[col] = tr[col]
			te_z[col] = te[col]
	tr_z.to_csv(fntr_z,index=False)
	te_z.to_csv(fnte_z,index=False)

def cz_standardize_data(fntr, fnte, fntr_z, fnte_z):
	'''
	Given training/testing set, calculate z' score.
	'''
	print 'cz_standardizing data...'
	print fntr
	tr = pd.read_csv(fntr)
	te = pd.read_csv(fnte)
	mn = tr.mean()
	sd = tr.std()
	col_list = tr.columns
	tr_z = pd.DataFrame(columns=col_list)
	te_z = pd.DataFrame(columns=col_list)
	for col in col_list:
		print col
		if col in labs:
			tr_z[col] = [my_z_socre(x, reference_range[col][0], reference_range[col][1], 
				mn[col], sd[col]) for x in tr[col]]
			te_z[col] = [my_z_socre(x, reference_range[col][0], reference_range[col][1], 
				mn[col], sd[col]) for x in te[col]]
		else:
			tr_z[col] = tr[col]
			te_z[col] = te[col]
	tr_z.to_csv(fntr_z,index=False)
	te_z.to_csv(fnte_z,index=False)

def my_z_socre(x, low, high, mn, sd):
	zl = (low - mn) / sd
	zh = (high - mn) / sd
	zx = (x - mn) / sd

	if zx >= zl and zx <= zh:
		return 0
	elif zx > zh:
		return zx - zh
	else:
		return zx - zl

def z_to_raw_data(fntr_z, fnte_z, mn, sd, fntr, fnte):
	print 'z_to_raw_data'
	tr_z = pd.read_csv(fntr_z)
	te_z = pd.read_csv(fnte_z)
	col_list = tr_z.columns
	tr = pd.DataFrame(columns=col_list)
	te = pd.DataFrame(columns=col_list)
	for col in col_list:
		if col in labs:
			tr[col] = [zscore_to_raw(x, mn[col], sd[col]) for x in tr_z[col]]
			te[col] = [zscore_to_raw(x, mn[col], sd[col]) for x in te_z[col]]
		else:
			tr[col] = tr_z[col]
			te[col] = te_z[col]
	tr.to_csv(fntr,index=False)
	te.to_csv(fnte,index=False)

def zscore_to_raw(x, mn, sd):
	return (x * sd) + mn

def break_time_one_hour_interval(fnin, fout):
	df = pd.read_csv(fnin)
	out_df = pd.DataFrame(columns=df.columns)
	out_idx = 0

	gp = df.groupby('sid')
	for sid, g in gp:
		print sid

		t = 0
		tdf = pd.DataFrame(columns=g.columns)
		tdf_idx = 0

		print 'shape[0]:', g.shape[0]
		g_idx = 0
		while g_idx < g.shape[0]:
			if g.iloc[g_idx]['timeindex'] >= t and g.iloc[g_idx]['timeindex'] < t + 60:
				tdf.loc[tdf_idx] = g.iloc[g_idx]
				tdf_idx += 1
				g_idx += 1
			else:
				if tdf.shape[0] > 0:
					mn = tdf.mean()
					mn['timeindex'] = t
					out_df.loc[out_idx] = mn
					out_idx += 1
				# else:
				# 	print 'check me', sid, t
				t += 60
				tdf = pd.DataFrame(columns=g.columns)
				tdf_idx = 0
		if tdf.shape[0] > 0:
			mn = tdf.mean()
			mn['timeindex'] = t
			out_df.loc[out_idx] = mn
			out_idx += 1
		# out_df.to_csv('tmp.csv',index=False)

	out_df.to_csv(fout, index=False)

def impute_by_interpolation_on_last12h(fin, fout, logfile='../data/seed2222/raw/extrapolation_log.txt'):
	print 'impute_by_interpolation_on_last12h'
	data = pd.read_csv(fin)# data.columns = cols
	grouped = data.groupby('sid')
	interped = pd.DataFrame(columns=data.columns)

	logstr = ''

	for sid, group in grouped:
		print sid
		# t = group.sort_values(by = 'timeindex', ascending = False)
		t = group.sort(columns='timeindex',ascending = False)
		
		lasttimeindex = t.iloc[0]['timeindex']

		if lasttimeindex < toffset:
			interped = interped.append(group)
			continue

		starttimeindex = lasttimeindex - toffset
		before_last12h_data = group[group['timeindex']<starttimeindex]

		imputed_last12 = pd.DataFrame(columns=data.columns)
		# imputed_pt = pd.DataFrame(columns=data.columns)
		for col in t:
			# print col
			imputed_col = []
			for i,v in t[col].iteritems():
				if not np.isnan(v):
				# print i,v
					curtimeindex = t.loc[i]['timeindex']
					if curtimeindex == lasttimeindex:
						imputed_col = interp_lab_last12h(group,col,lasttimeindex)
						logstr += '%d_%s: interped\n'%(sid,col)
						# interp
					elif curtimeindex < lasttimeindex - toffset:
						# hold
						n = group[group['timeindex']>=(lasttimeindex-toffset)].shape[0]
						# n = group.shape[0]
						imputed_col = [v] * n
						logstr += '%d_%s: hold\n'%(sid,col)
					else:
						# hold forward and backward
						imputed_col = hold_fb_lab_last12h(group,col,lasttimeindex,curtimeindex,v)
						logstr += '%d_%s: hold_forward_and_backward\n'%(sid,col)
					break
			# print imputed_col
			if len(imputed_col) == 0:
				# imputed_col = [np.nan] * group.shape[0]
				imputed_col = [np.nan] * group[group['timeindex']>=(lasttimeindex-toffset)].shape[0]
			imputed_last12[col] = imputed_col
			# imputed_pt[col] = imputed_col

		imputed_pt = before_last12h_data.append(imputed_last12)
		interped = interped.append(imputed_pt)
	interped.to_csv(fout,index=False)
	fn = open(logfile,'w')
	fn.write(logstr)
	fn.close()

def interp_lab_last12h(df,col,lasttimeindex):
	starttimeindex = lasttimeindex - toffset
	# print starttimeindex
	dx = df[df['timeindex']>=starttimeindex]
	xvals = dx['timeindex']

	# dx = df
	# xvals = dx['timeindex']

	# print xvals

	x = []
	y = []
	for i in range(dx.shape[0]):
		val = dx.iloc[i][col]
		ti = dx.iloc[i]['timeindex']

		if not np.isnan(val):
			x.append(ti)
			y.append(val)

	return np.interp(xvals,x,y)
# test
# df = pd.read_csv('../data/seed2222/raw/train_fold0.csv')
# d21 = df[df['sid']==3771]
# d21.iloc[15]['Art_CO2'] = 17
# print interp_lab_last12h(d21,'Art_CO2',1117)
# print interp_lab_last12h(d21,'readmit',6807)


# df = pd.read_csv('../data/seed2222/raw/test_fold0.csv')
# d21 = df[df['sid']==21]
# d21.iloc[15]['Art_CO2'] = 17
# print interp_lab_last12h(d21,'Art_CO2',1117)
# print interp_lab_last12h(d21,'timeindex',1117)

def hold_fb_lab_last12h(df,col,lasttimeindex,curtimeindex,v):
	starttimeindex = lasttimeindex - toffset
	dx = df[df['timeindex']>=starttimeindex]
	xvals = dx['timeindex']
	# dx = df
	# xvals = dx['timeindex']

	x = []
	y = []
	for i in range(dx.shape[0]):
		val = dx.iloc[i][col]
		ti = dx.iloc[i]['timeindex']

		if ti <= curtimeindex and not np.isnan(val):
			x.append(ti)
			y.append(val)
		if ti > curtimeindex:
			x.append(ti)
			y.append(v)

	return np.interp(xvals,x,y)

def get_last_measurements(fin,fout):
      data = pd.read_csv(fin)
      # data.columns = cols
      grouped = data.groupby('sid')

      lm = pd.DataFrame(columns=data.columns)
      for sid, group in grouped:
      	# t = group.sort_values(by = 'timeindex', ascending = False)
            t = group.sort(columns='timeindex',ascending = False)
            s = pd.Series(index=data.columns)
            for col in t:
                  for i,v in t[col].iteritems():
                        if not np.isnan(v):
                              # print i,v
                              s[col] = v
                              break
            lm = lm.append(s,ignore_index=True)
      lm.to_csv(fout,index=False)
# test
# df = pd.read_csv('../data/seed2222/raw/test_fold0.csv')
# d21 = df[df['sid']==112]
# d21.iloc[15]['Art_CO2'] = 17
# print hold_fb_lab_last12h(d21,'RSBIRate',1117,97,0)
# print hold_fb_lab_last12h(d21,'RESP',1158,993,18)


# for i in range(5):
# 	impute_by_interpolation_on_last12h(
# 		'../data/seed2222/raw/test_fold%d.csv'%i, 
# 		'../data/seed2222/raw/interp/alltime/test_fold%d.csv'%i, 
# 		'../data/seed2222/raw/interp/alltime/extrapolation_log_test_fold%d.txt'%i)
# 	impute_by_interpolation_on_last12h(
# 		'../data/seed2222/raw/train_fold%d.csv'%i, 
# 		'../data/seed2222/raw/interp/alltime/train_fold%d.csv'%i, 
# 		'../data/seed2222/raw/interp/alltime/extrapolation_log_train_fold%d.txt'%i)

# split_test_by_patient('../data/seed2222/raw/interp',
# 	'../data/seed2222/raw/interp/mice/imputing')

# for i in range(5):
# 	ori_tr = pd.read_csv('../data/seed2222/raw/train_fold%d.csv'%i)
# 	z_to_raw_data('../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d_z.csv'%i,
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d_z.csv'%i,
# 		ori_tr.mean(), ori_tr.std(),
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i)
# 	cz_standardize_data('../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i,
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d_cz.csv'%i,
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d_cz.csv'%i)

# (one hour interval)
# for i in range(5):
# 	ori_tr = pd.read_csv('../data/seed2222_one_hour_interval/raw/train_fold%d.csv'%i)
# 	z_to_raw_data('../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d_z.csv'%i,
# 		'../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d_z.csv'%i,
# 		ori_tr.mean(), ori_tr.std(),
# 		'../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
# 		'../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i)
# 	cz_standardize_data('../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d.csv'%i,
# 		'../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d.csv'%i,
# 		'../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/train_fold%d_cz.csv'%i,
# 		'../data/seed2222_one_hour_interval/raw/standardize_z_mice/mp0.5_mc0.6/dataset/test_fold%d_cz.csv'%i)


# split_test_by_patient('../data/seed2222/raw',
# 	'../data/seed2222/raw/standardize_z_mice/imputing',
# 	'_z')

# cu.checkAndCreate('../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6')
# for i in range(5):
# 	standardize_data(
# 		'../data/seed2222/raw/train_fold%d.csv'%i,
# 		'../data/seed2222/raw/test_fold%d.csv'%i,
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/train_fold%d_z.csv'%i,
# 		'../data/seed2222/raw/standardize_z_mice/mp0.5_mc0.6/test_fold%d_z.csv'%i)