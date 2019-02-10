"""
Feature selection with RFE.
Created by yxue - 05-18-2017
"""

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import os

# cdn = '../data/seed2222/raw/interp/mice/mp0.5_mc0.6/last_measures'
# cdn = '../data/seed2222/raw/interp/mean/last_measures'

def rfe(cdn, n, feature_type, c, p, bl):
	print 'rfe...'

	ranks = {}
	selected_cols = []

	for k in range(5):
		if feature_type is None:
			train  = pd.read_csv('%s/dataset/train_fold%d.csv'%(cdn,k))
			test = pd.read_csv('%s/dataset/test_fold%d.csv'%(cdn,k))
		else:
			train = pd.read_csv('%s/dataset/train_fold%d_%s.csv'%(cdn,k,feature_type))
			test = pd.read_csv('%s/dataset/test_fold%d_%s.csv'%(cdn,k,feature_type))

		x_train = train.drop(['readmit','sid'],axis=1)
		y_train = train['readmit']

		cols = x_train.columns

		logreg = LogisticRegression(penalty=p, C=c, class_weight=bl)
		selector = RFE(logreg,n,step=1)
		selector = selector.fit(x_train, y_train)
		# selector = selector.fit(X, y)
		ranks[k] = selector.ranking_
		# print selector.support_
		# print selector.ranking_

	for i in range(len(ranks[0])):
		if ranks[0][i] == 1 and ranks[1][i] == 1 and ranks[2][i] == 1 and ranks[3][i] == 1 and ranks[4][i] == 1:
			selected_cols.append(cols[i])

	# print selected_cols

	checkAndCreate('%s/rfe_cols/%s'%(cdn,feature_type))
	with open('%s/rfe_cols/%s/best%d_c%s_%s_%s'%(cdn,feature_type,n,str(c),p,bl),'wb') as f:
		pickle.dump(selected_cols,f)

	return selected_cols


def checkAndCreate(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)
