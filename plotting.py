import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# import plotly
# import plotly.tools as tls
# import plotly.plotly as py
# py.sign_in('jay.xue', '5k7xkxhujs')
import preprocessing as pp

labs = ['Creatinine', 'BUN', 'BUNtoCr', 
'urineByHrByWeight', 'eGFR', 'AST', 'ALT', 'TBili', 'DBili', 'Albumin', 'tProtein', 
'ASTtoALT', 'HCT', 'Hgb', 'INR', 'Platelets', 'PT', 'PTT', 'RBC', 'WBC', 'RESP', 
'mSaO2', 'PaO2toFiO2', 'MinuteVent', 'DeliveredTidalVolume', 'FiO2Set', 'PEEPSet', 
'PIP', 'PlateauPres', 'RAW', 'RSBI', 'RSBIRate', 'mSBP', 'mDBP', 'mMAP', 'CV_HR', 
'mCrdIndx', 'mCVP', 'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_PaO2', 'Art_pH', 'Na', 
'K', 'Cl', 'Glucose', 'Ca', 'Mg', 'IonCa', 'Lactate', 'GCS', 'temp', 'Age']

def plot_ori_vs_imp(fn_ori, fn_imp, fout, byPt=True, patient_idlist=None, mn=None, sd=None):
	data_ori = pd.read_csv(fn_ori)
	data_imp = pd.read_csv(fn_imp)
	if byPt:
		if patient_idlist == None:
			patient_idlist = data_ori['sid'].unique().tolist()
			print patient_idlist
			print len(patient_idlist)
		for ptid in patient_idlist:
		# for ptid in [138]:
			gp_ori = data_ori.groupby('sid')
			gp_imp = data_imp.groupby('sid')

			pt_ori = gp_ori.get_group(float(ptid))
			pt_imp = gp_imp.get_group(float(ptid))

			X = pt_ori['timeindex'].tolist()
			# print X
			# for col in pt_ori.columns:
			for col in labs:
				Y = np.array([])
				c = np.array([])
				plt.clf()
			# for col in ['Creatinine.standardized']:
				for i in pt_ori.index.values:
					if np.isnan(pt_ori[col][i]):
						Y = np.append(Y,pt_imp[col][i])
						c = np.append(c,'r')
					else:
						Y = np.append(Y,pt_ori[col][i])
						c = np.append(c,'g')
				
				fig = plt.figure()
				fig.suptitle(col, fontsize=14, fontweight='bold')
				ax = fig.add_subplot(111)
				
				ax.set_title('mn: %.2f std: %.2f reference range: (%s,%s)'
					%(mn[col],sd[col],pp.reference_range[col][0],pp.reference_range[col][1]))
				maxY = max(Y)
				minY = min(Y)
				if minY > 0:
					ylml = minY*0.9-1
				else:
					ylml = minY*1.1-1
				if maxY > 0:
					ylmh = maxY*1.1+1
				else:
					ylmh = maxY*0.9+1
				ax.set_ylim(ylml,ylmh)
				# ax.scatter(X,Y,c=c)
				ax.plot(X,Y)
				# plt.text(3, 8, 'boxed italics text in data coords')
				# plt.title(col)
				# plt.scatter(X,Y,c=c)
				# plt.plot(X,Y)
				
				# plt.show()
				# if not os.path.exists('../data/plot_of_imputes_values/%d/'%ptid):
				# 	os.makedirs('../data/plot_of_imputes_values/%d/'%ptid)
				# plt.savefig('../data/plot_of_imputes_values/%d/%d_%s.png' % (ptid, ptid, col))
				if not os.path.exists('%s/%s/'%(fout,ptid)):
					os.makedirs('%s/%s/'%(fout,ptid))
				plt.savefig('%s/%s/%s_%s.png' % (fout, ptid, ptid, col))
				# exit(0)
	else:
		X = data_ori.index.values
		# print X
		for col in data_ori.columns:
			Y = np.array([])
			Y_x = np.array([])
			I = np.array([])
			I_x = np.array([])
			c = np.array([])
			# plt.clf()
			# ori.plot.clf()
		# for col in ['Creatinine.standardized']:
			for i in X:
				if np.isnan(data_ori[col][i]):
					I = np.append(I,data_imp[col][i])
					I_x = np.append(I_x,i)
					# c = np.append(c,'r')
				else:
					Y = np.append(Y,data_ori[col][i])
					Y_x = np.append(Y_x,i)
					# c = np.append(c,'g')
			fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=True)
			fig.append_trace({'x': Y_x, 'y': Y, 'type': 'scatter', 'name': 'ori'}, 1, 1)
			fig.append_trace({'x': I_x, 'y': I, 'type': 'scatter', 'name': 'imp'}, 1, 1)
			fig.append_trace({'x': X, 'y': data_imp[col], 'type': 'scatter', 'name': 'ori+imp'}, 1, 1)
			fn = '%s.png' % (col)
			py.plot(fig, filename=fn)
			
			# ori = pd.Series(Y)
			# ori.plot.kde()
			
			# plt.plot(Y_x, Y, color='blue', linewidth=2.5, linestyle="-")
			# plt.plot(I_x, I, color='red', linewidth=2.5, linestyle="-")

			# plt.scatter(X,Y,c=c)
			# plt.show()
			# if not os.path.exists('../data/plot_of_imputes_values/all/'):
			# 	os.makedirs('../data/plot_of_imputes_values/all/')
			# ori.plot.savefig('../data/plot_of_imputes_values/all/%s.png' % (col))
			# exit(0)



# plot_ori_vs_imp('../data/toMICEdata/alldata_readmit_last12h_X_train_fold1.csv', '../data/train_mice/last12h_pmm_mp0.4_mc0.6_fold1.csv', True)
# plot_ori_vs_imp('../data/toMICEdata/alldata_readmit_last12h_X_train_fold1.csv', '../data/train_mice/last12h_pmm_mp0.4_mc0.6_fold1.csv', byPt=False)
# plot_ori_vs_imp('../data/toMICEdata/alldata_readmit_last12h_X_train_fold1.csv', '../data/last12hmeandata/alldata_readmit_last12h_mean_train_fold1.csv', byPt=True)
# plot_ori_vs_imp('../data/seed2222/z/train_fold2.csv', '../data/seed2222/z/mice/mp0.4_mc0.6/train_fold2.csv', byPt=True, patient_idlist=[61,184,12940])
# plot_ori_vs_imp('../data/seed2222/z/test_fold2.csv', '../data/seed2222/z/mice/mp0.4_mc0.6/test_fold2.csv', byPt=True, patient_idlist=[995,899,14321])

# Creatinine.standardized
# BUN.standardized
# BUNtoCr.standardized