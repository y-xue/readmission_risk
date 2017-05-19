# java -cp moss.jar moss.Miner -inelist -onelist -m1 -s0.001 last12h/data/mimic.nel last12h/data/mimic_m1_s0.001.out last12h/data/mimic_m1_s0.001.ids &> log_m1_s0.001
# java -cp moss.jar moss.Miner -inelist -onelist -m1 -s0.003 data/mimic.nel data/mimic_m1_s0.003.out data/mimic_m1_s0.003.ids &> log_m1_s0.003
# java -cp moss.jar moss.Miner -inelist -onelist -m1 -s0.005 data/last12h_mean_fold0.nel data/last12h_mean_fold0_m1_s0.005.out data/last12h_mean_fold0_m1_s0.005.ids &> log_last12h_mean_fold0_m1_s0.005

# To generate the graph file using interpolation at regular time interval:
# import nel_graph_gen_interpolation as nggi
# x = nggi.NelGraphGenInterpolation()
# x.scan_csv_interpolation("../data/alldata.csv", "../data/mimic.nel", "../data/mimic.node")

# import nel_graph_gen_interpolation as nggi
# x = nggi.NelGraphGenInterpolation()
# for i in range(5):
# 	x.scan_csv_interpolation("../data/alldata_last12h_mean_fold%d.csv"%i, "../data/alldata_last12h_mean_fold%d_mimic.nel"%i, "../data/alldata_last12h_mean_fold%d_mimic.node"%i)


import mimic_tensor as mt
import mimic_data as md
import numpy as np
import mimic_classifying as mc
import subgraph_mining as sm
# import mimic_clustering as mclu
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression

cdn = '../data/mean'

def write_pt_sg_before_filter_pt(ptsg, ptwd, sgs, pt, gt, isg, s, i):
	with open('%s/isg%d/pt_sg_w_%s/ptsg_fold%d'%(cdn,isg,s,i),'wb') as f:
		pickle.dump(ptsg,f)
	with open('%s/isg%d/pt_sg_w_%s/ptwd_fold%d'%(cdn,isg,s,i),'wb') as f:
		pickle.dump(ptwd,f)
	with open('%s/isg%d/pt_sg_w_%s/sgs_fold%d'%(cdn,isg,s,i),'wb') as f:
		pickle.dump(sgs,f)
	with open('%s/isg%d/pt_sg_w_%s/pt_fold%d'%(cdn,isg,s,i),'wb') as f:
		pickle.dump(pt,f)
	with open('%s/isg%d/pt_sg_w_%s/gt_fold%d'%(cdn,isg,s,i),'wb') as f:
		pickle.dump(gt,f)

def gen_pt_sg_files(isg, s, i):
	print 'gen_pt_sg_files:'
	print isg, s, i
	mt.pt_sg_w_tensor_gen("../data/alldata_last12h_mean_fold%d_mimic.nel"%i, "../data/alldata_last12h_mean_fold%d_mimic.node"%i, "../data/test/last12h_mean_mimic_m1_s0.%s_fold%d.out"%(s,i), "../data/test/last12h_mean_mimic_m1_s0.%s_fold%d.ids"%(s,i), "../data/mean/isg%d/pt_sg_w_%s/mimic.tensor_fold%d"%(isg,s,i), "../data/mean/isg%d/pt_sg_w_%s/mimic_pt_sg.spmat_fold%d"%(isg,s,i), "../data/mean/isg%d/pt_sg_w_%s/mimic_pt_w.spmat_fold%d"%(isg,s,i), "../data/mean/isg%d/pt_sg_w_%s/mimic.ptid_fold%d"%(isg,s,i), "../data/mean/isg%d/pt_sg_w_%s/mimic.sgtid_fold%d"%(isg,s,i), "../data/mean/isg%d/pt_sg_w_%s/mimic.sgstr_fold%d"%(isg,s,i), "../data/mean/isg%d/pt_sg_w_%s/mimic.ntid_fold%d"%(isg,s,i), isg=isg)

	# perform supervised learning
	print 0
	ptsg = md.read_pt_sg_mat(isg,s,i)
	print 1
	ptwd = md.read_pt_wd_mat(isg,s,i)
	print 2
	sgs = md.read_sgs(isg,s,i)
	print 3
	spt = md.read_sparse_tensor(isg,s,i)
	print 4
	(hiso, hsgstr, hsgc, hsgsize) = md.sg_subiso(sgs)
	print 5
	# print ptsg.shape
	(ptsg, sgs, sptsel) = md.filter_sg(ptsg, hiso, sgs, spt=spt)
	# print ptsg.shape
	df_ptsg = pd.DataFrame(ptsg)
	df_ptsg.to_csv('%s/isg%d/df_ptsg_%s_fold%d.csv'%(cdn,isg,s,i), index=False)

	print 6
	# ptsg = np.hstack((ptsg, ptwd))
	gt = md.read_pt_gt(isg,s,i)
	print 7
	pt = md.read_pt_lab(isg,s,i)
	print 8
	write_pt_sg_before_filter_pt(ptsg, ptwd, sgs, pt, gt, isg, s, i)

def classify(isg, s, ntestth, i):
	print 'classify:'
	print 'isg, s, ntestth:', isg, s, ntestth
	ptsg = pickle.load(open('%s/isg%d/pt_sg_w_%s/ptsg_fold%d'%(cdn,isg,s,i),'rb'))
	ptwd = pickle.load(open('%s/isg%d/pt_sg_w_%s/ptwd_fold%d'%(cdn,isg,s,i),'rb'))
	sgs = pickle.load(open('%s/isg%d/pt_sg_w_%s/sgs_fold%d'%(cdn,isg,s,i),'rb'))
	pt = pickle.load(open('%s/isg%d/pt_sg_w_%s/pt_fold%d'%(cdn,isg,s,i),'rb'))
	gt = pickle.load(open('%s/isg%d/pt_sg_w_%s/gt_fold%d'%(cdn,isg,s,i),'rb'))
	# (ptsg, ptwd, pt, gt, sptr) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = 5, spt=sptsel)
	(ptsg, ptwd, pt, gt) = md.filter_pt(ptsg, ptwd, sgs, pt, gt, ntest_th = ntestth)

	# (ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te(ptsg, gt, pt)
	(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te_by_ptids(ptsg, gt, pt, "../data/train/alldata_last12h_X_train_fold%d.csv"%i, "../data/test/alldata_last12h_X_test_fold%d.csv"%i, "%s/isg%d/pt_sg_w_%s/mimic.ptid_fold%d" % (cdn,isg,s,i))

	clf = LogisticRegression(penalty='l1', class_weight='balanced')
	res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l1_bl_fold%d'%(cdn,isg,s,ntestth,i),'wb') as f:
		pickle.dump(res,f)

	clf = LogisticRegression(penalty='l2', class_weight='balanced')
	res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l2_bl_fold%d'%(cdn,isg,s,ntestth,i),'wb') as f:
		pickle.dump(res,f)

	clf = LogisticRegression(penalty='l1', class_weight=None)
	res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l1_nb_fold%d'%(cdn,isg,s,ntestth,i),'wb') as f:
		pickle.dump(res,f)

	clf = LogisticRegression(penalty='l2', class_weight=None)
	res = mc.directClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, clf=clf)
	with open('%s/isg%d/pt_sg_w_%s/res_nt%d_l2_nb_fold%d'%(cdn,isg,s,ntestth,i),'wb') as f:
		pickle.dump(res,f)

def print_res_auc(isg, s, ntestth, i):
	output = ''
	print '(isg=%d,s=%s,ntestth=%d,fold=%d):'%(isg,s,ntestth,i)
	for pl in ['l1','l2']:
		for cw in ['bl', 'nb']:
			res = pickle.load(open('%s/isg%d/pt_sg_w_%s/res_nt%d_%s_%s_fold%d'%(cdn,isg,s,ntestth,pl,cw,i),'r'))
			print pl,cw
			print res['auc_te']
			print 
			output += '%s %s\n'%(pl,cw)
			output += '%.03f\n\n'%res['auc_te']
	if not os.path.exists('%s/isg%d/res_fold%d'%(cdn,isg,i)):
		os.makedirs('%s/isg%d/res_fold%d'%(cdn,isg,i))
	fn = open('%s/isg%d/res_fold%d/%s_nt%d.txt'%(cdn,isg,i,s,ntestth),'w')
	fn.write(output)
	fn.close()

for s in ['001','002','005','006','008','009']:
	print 's = %s'%s
	if s == '006':
		continue
	for i in range(5):
		print 'fold %d'%i
		sm.sub_mining_ptbypt('../data/train/last12h_mean_train_fold%d.csv'%i, '../data/test/last12h_mean_test_fold%d.csv'%i, i, s)


for s in ['001','002','005','006','008','009']:
	for i in range(5):
		for isg in [0,3,1,2]:
			# if i != 0:
			gen_pt_sg_files(isg, s, i)
			for ntestth in [2,5]:
				classify(isg,s,ntestth,i)
				print_res_auc(isg,s,ntestth,i)

# for s in ['001','002','006']:
# 	for i in range(5):
# 		for isg in [0,3,1,2]:
# 			# if i != 0:
# 			gen_pt_sg_files(isg, s, i)
# 			for ntestth in [2,5]:
# 				classify(isg,s,ntestth,i)
# 				print_res_auc(isg,s,ntestth,i)



# for isg in [0,1]:
# 	for s in ['001','002','003','004','005','006','008','009','01','011']:
# 		for ntestth in [2,5]:
# 			print_res_auc(isg, s, ntestth)


# print 15
# mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te, nc=100, norm='l2')
# (ptwd_tr, ptwd_te, gt_tr, gt_te, pt_tr, pt_te) = md.split_tr_te(ptwd, gt, pt)
# print 16
# mc.directClassify(ptwd_tr, ptwd_te, gt_tr, gt_te)



# print 16
# (pt_olab_tr, pt_olab_te, colns) = md.read_pt_olab_tr_te(pt_tr, pt_te, fn='data/pt_olab_cnt_last_12h.csv')
# print 17
# (saps_tr, saps_te) = md.read_pt_saps_tr_te(pt_tr, pt_te)
# print 18
# df_spt = pd.DataFrame(sptr); df_spt.iloc[:,[0,1,2]] = df_spt.iloc[:,[0,1,2]].astype(int); df_spt.to_csv('data/pt_sg_w/mimicf.tensor', header=False, index=False)
# print 19
# mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te, nc=100, norm='l2')
# print 20
# mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te, nc=150, header="C1", fngrp='result/ptsg_nmf150_clf_grp.txt', sgs=sgs)
# print 21
# # mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te, nc=80, header="C1", fngrp='result/ptsg_nmf80_clf_grp.txt', sgs=sgwds)
# # print 22
# mc.directClassify(np.hstack((ptsg_tr, .5*md.binary(ptwd_tr))), np.hstack((ptsg_te, .5*md.binary(ptwd_te))), gt_tr, gt_te)
# print 23

# mc.nmfClassify(ptwd_tr, ptwd_te, gt_tr, gt_te, pt_tr, pt_te, nc=50, norm='l2', fnfmt='/Users/XueY/Desktop/summer project/code/temporal-trends/code/ptwd_nmf/nmf_%s.pik')
# print 24
# mc.nmfClassify(ptsg_tr, ptsg_te, gt_tr, gt_te, pt_tr, pt_te, nc=60, fnfmt='/Users/XueY/Desktop/summer project/code/temporal-trends/code/ptsg_nmf_0.003_nndsvd_sparsity/nmf_%d.pik', fngrp='result/ptsg_nmf60_clf_grp.txt', sgs=sgs)
# print 25

# # mc.directClassify(saps_tr.loc[:,['lnSAPSII']], saps_te.loc[:,['lnSAPSII']], gt_tr, gt_te, fnroc='result/saps_clf.csv', mean_impute=True)
# # print 26

# # mclu.visualizePCA(ptsg_tr, gt_tr, 'training pt-sg PCA', 'figures/pt_sg_training_pca.pdf')
