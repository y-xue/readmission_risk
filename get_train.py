import pandas as pd
from numpy import *
import csv

toffset = 720
tcmax = 6 
tu = 60*2 # 2 hrs intervals, 12 hrs

def combine_imputed_tr_te():
	for i in range(5):
		xtrain = pd.read_csv('../data/train/alldata_last12h_X_train_fold0.csv')
		ytrain = pd.Series.from_csv('../data/train/alldata_last12h_y_train_fold0.csv')
		xtest = pd.read_csv('../data/test/alldata_last12h_X_test_fold0.csv')
		ytest = pd.Series.from_csv('../data/test/alldata_last12h_y_test_fold0.csv')

		xtrain['died'] = ytrain
		xtest['died'] = ytest

		cols = xtrain.columns.tolist()
		cols = cols[-1:] + cols[:-1]
		xtrain = xtrain[cols]
		xtest = xtest[cols]

		data = xtrain.append(xtest)
		print data.columns

		data.to_csv('../data/alldata_last12h_mean_fold%d'%i,index=False)

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

def interpolating(ptarr, sid):
	hlab = {}
	(rows, cols) = ptarr.shape
	tidx = ptarr[:,0]

	if not has_last_12h(tidx):
		print 'here'
		return None
	thridx = tidx[-1] - 720 + tu * (arange(tcmax) + 1)
	thridx = thridx.reshape(tcmax, 1); iptarr = thridx

	for ci in range(1, cols-2):
		y = ptarr[:,ci]
		yi = interp(thridx, tidx, y)
		iptarr = hstack((iptarr, yi))
	y = ptarr[:,cols-2]
	yi = repeat_last(thridx, tidx, y)
	iptarr = hstack((iptarr, yi))

	y = ptarr[:,cols-1]
	yi = repeat_last(thridx, tidx, y)
	iptarr = hstack((iptarr, yi))

	return iptarr

def scan_csv_interpolation(fnin):
	fin = open(fnin,'r')
	freader = csv.reader(fin,delimiter=",",quotechar="\"")
	lcnt = 0; current_sid = ""; ptarr = None; gcnt = 0; current_mort = 0;

	hv = {}
	for row in freader:
		lcnt += 1
		if lcnt == 1:
			vns = ['died', 'sid', 'timeindex',
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
				interp_arr = interpolating(ptarr, current_sid)
				# print type(interp_arr)
				# print interp_arr
				# print interp_arr.shape
				current_sid = sid; ptarr = None
				# exit(0)
			arow = array([time, hv['Antiarrhythmic_m'], hv['Anticoagulant_m'], hv['Antiplatelet_m'], hv['Benzodiazepine_m'], hv['beta_Blocking_m'], hv['Calcium_channel_blocking_m'], hv['Diuretic_m'], hv['Hemostatic_m'], hv['Inotropic_m'], hv['Insulin_m'], hv['Nondepolarizing_m'], hv['sedatives_m'], hv['Somatostatin_preparation_m'], hv['Sympathomimetic_m'], hv['Thrombolytic_m'], hv['Vasodilating_m'],hv['AIDS_p'], hv['HemMalig_p'], hv['MetCarcinoma_p'], hv['Creatinine_n'], hv['BUN_n'], hv['BUNtoCr_n'], hv['urineByHrByWeight_n'], hv['eGFR_n'], hv['AST_n'], hv['ALT_n'], hv['TBili_n'], hv['DBili_n'], hv['Albumin_n'], hv['tProtein_n'], hv['ASTtoALT_n'], hv['HCT_n'], hv['Hgb_n'], hv['INR_n'], hv['Platelets_n'], hv['PT_n'], hv['PTT_n'], hv['RBC_n'], hv['WBC_n'], hv['RESP_n'], hv['mSaO2_n'], hv['PaO2toFiO2_n'], hv['MinuteVent_n'], hv['DeliveredTidalVolume_n'], hv['FiO2Set_n'], hv['PEEPSet_n'], hv['PIP_n'], hv['RSBI_n'], hv['RSBIRate_n'], hv['RAW_n'], hv['PlateauPres_n'], hv['mSBP_n'], hv['mDBP_n'], hv['mMAP_n'], hv['CV_HR_n'], hv['mCrdIndx_n'], hv['mCVP_n'], hv['Art_BE_n'], hv['Art_CO2_n'], hv['Art_PaCO2_n'], hv['Art_PaO2_n'], hv['Art_pH_n'], hv['Na_n'], hv['K_n'], hv['Cl_n'], hv['Glucose_n'], hv['Ca_n'], hv['Mg_n'], hv['IonCa_n'], hv['Lactate_n'], hv['GCS_n'], hv['temp_n'], hv['medtype.label'], hv['location.label']])
			# print len(arow)
			arow = arow.reshape(1, arow.size)
			if ptarr == None:
				ptarr = arow
			else:
				ptarr = vstack((ptarr,arow))
			current_sid = sid; current_mort = int(hv['died'])

	interp_arr = interpolating(ptarr, current_sid)

def split_test_by_patient():
	for i in range(5):
		test = pd.read_csv('../data/toMICEdata/alldata_readmit_last12h_X_test_fold%d.csv'%(i))
		gp = test.groupby('sid')
		j = 0
		for sid, group in gp:
			group.to_csv('../data/toMICEdata/fold%d/%d.csv'%(i,j),index=False)
			j += 1

# split_test_by_patient()
# scan_csv_interpolation('../data/train/last12h_mean_train_fold0.csv')

# combine_imputed_tr_te()

def add_outcome_col(minp,minc):
	for i in range(5):
		xtrain = pd.read_csv('../data/train/last12h_pmm_X_mp%s_mc%s_fold%d.csv'%(str(minp),str(minc),i))
		# xtrain = pd.read_csv('../data/train/last12h_mean_X_train_fold%d.csv'%(i))
		ytrain = pd.Series.from_csv('../data/toMICEdata/alldata_readmit_last12h_y_train_fold%d.csv'%i)
		xtrain['readmit'] = ytrain
		cols = xtrain.columns.tolist()
		cols = cols[-1:] + cols[:-1]
		xtrain = xtrain[cols]
		xtrain.to_csv('../data/train/last12h_pmm_mp%s_mc%s_fold%d.csv'%(str(minp),str(minc),i),index=False)

		# xtest = pd.read_csv('../data/test/last12h_mean_X_test_fold%d.csv'%(i))
		
		xtest = pd.read_csv('../data/test/fold%d/fold%d_mp%s_mc%s.csv'%(i,i,str(minp),str(minc)))
		ytest = pd.Series.from_csv('../data/toMICEdata/alldata_readmit_last12h_y_test_fold%d.csv'%i)
		xtest['readmit'] = ytest
		cols = xtest.columns.tolist()
		cols = cols[-1:] + cols[:-1]
		xtest = xtest[cols]
		xtest.to_csv('../data/test/fold%d_mp%s_mc%s.csv'%(i,str(minp),str(minc)),index=False)

mp_list = ["0.4","0.5","0.65","0"]
mc_list = ["0.1","0.3","0.4","0.6"]
for mp in mp_list:
	for mc in mc_list:
		add_outcome_col(mp,mc)



# df = pd.read_csv('../data/alldata_readmit.csv')
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

# from sklearn.model_selection import KFold
# df = pd.read_csv('../data/alldata_readmit_last12h.csv')
# gp = df.groupby('sid')
# kf = KFold(len(gp), n_folds=5)
# j = 0
# for train_idx, test_idx in kf:
# 	trainset = pd.DataFrame(columns=df.columns)
# 	testset = pd.DataFrame(columns=df.columns)
# 	i = 0
# 	for sid,g in gp:
# 		if i in train_idx:
# 			trainset = trainset.append(g)
# 		elif i in test_idx:
# 			testset = testset.append(g)
# 		i += 1
# 	trainset.to_csv('../data/alldata_readmit_last12h_train_fold%d.csv'%j,index=False)
# 	testset.to_csv('../data/alldata_readmit_last12h_test_fold%d.csv'%j,index=False)
# 	j += 1

# for i in range(5):
# 	train = pd.read_csv('../data/alldata_readmit_last12h_train_fold%d.csv'%i)
# 	test = pd.read_csv('../data/alldata_readmit_last12h_test_fold%d.csv'%i)
# 	X_train = train.drop(['readmit'],axis=1)
# 	y_train = train['readmit']
# 	X_test = test.drop(['readmit'],axis=1)
# 	y_test = test['readmit']
# 	X_train.to_csv('../data/alldata_readmit_last12h_X_train_fold%d.csv'%i,index=False)
# 	X_test.to_csv('../data/alldata_readmit_last12h_X_test_fold%d.csv'%i,index=False)
# 	y_train.to_csv('../data/alldata_readmit_last12h_y_train_fold%d.csv'%i)
# 	y_test.to_csv('../data/alldata_readmit_last12h_y_test_fold%d.csv'%i)

