import re
import pandas as pd

def f(x):
	if x > 0:
		return 1
	else:
		return  0

def cnt_sg(fn):
	df = pd.read_csv(fn)
	cnt_list = df.sum(axis=0)
	print
	print 'len(cnt_list):', len(cnt_list)
	print 'used sgs:', len(filter(lambda x:x>0,cnt_list))

def cnt_sg_contains_0(fn='../data/mimic_m1_s0.001.out'):
	f = open(fn,'r')
	cnt = 0
	found_before = False
	for line in f:
		if re.search(r'g \d',line):
			found_before = False
		if found_before:
			continue
		if re.search(r'n \d .*_0',line):
			cnt += 1
			found_before = True
	return cnt

def read_sgs(isg, s,fn=None):
    if fn==None:
        fn='../data/isg%d/pt_sg_w_%s/mimic.sgstr'%(isg,s)
    sgs = []
    f = open(fn)
    for ln in f:
        ln = ln.rstrip(" \n")
        sgs.append(ln)
    f.close()
    return sgs;

def cnt_hisg_contains_0():
	output = ''
	for isg in [0,3]:
		output += 'isg %d:\n'%isg
		for s in ['001','002','003','004','005','006','008','009','01','011']:
			output += 'freq_t %s:\n'%s
			for i in range(5):
				output += 'fold %d: '%i
				sgs = read_sgs(isg,s,fn='%s/isg%d/pt_sg_w/mimic.sgstr_%s_fold%d'%(cdn,isg,s,i))

				cnt = 0
				for i in range(len(sgs)):
					sg = sgs[i]
					sgrow = sg.split('\t')
					if re.search(r'0', sgrow[2]):
						cnt += 1
				output += '%d; '%cnt
			output += '\n'
		output += '\n'
	fn = open('%s/cnt_hisg_contains_0.txt'%cdn,'w')
	fn.write(output)
	fn.close()

# cdn = '../data/raw_shuffled'
# cdn = '../data/mean'
# cdn = '../data/mice/mp0.4_mc0.6'
import pickle
import os
cdn = '../data/seed2222/z/mice/mp0.4_mc0.6'
def cnt_sg_in_ptsg():
	output = ''
	for isg in [0,3]:
		output += 'isg %d:\n'%isg
		for freq_t in ['001','002','003','004','005','006','008','009','01','011']:
			output += 'freq_t %s:\n'%freq_t
			for i in range(5):
				output += 'fold %d: '%i
				if os.path.isfile('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(cdn,isg,freq_t,i)):
					ptsg = pickle.load(open('%s/isg%d/pt_sg_w/ptsg_%s_fold%d'%(cdn,isg,freq_t,i),'rb'))
					output += '%d; '%ptsg.shape[1]
				else:
					output += 'NA'
			output += '\n'
		output += '\n'
	fn = open('%s/ptsg_cnt.txt'%cdn,'w')
	fn.write(output)
	fn.close()
cnt_sg_in_ptsg()
# cnt_hisg_contains_0()


# for i in range(5):
# 	print 'fold%d'%i
# 	for s in ['006']:
# 		print s
# 		for isg in [0,1,2,3]:
# 			print isg
# 			cnt_sg('../data/mean/isg%d/df_ptsg_%s_fold%d.csv'%(isg,s,i))
			# c,r = cnt_hisg_contains_0(read_sgs(isg,s,fn='../data/mean/isg%d/pt_sg_w_%s/mimic.sgstr_fold%d'%(isg,s,i)))
			# print s
			# print c,r
			# print

# for s in ['001','002','003','004','005','006','008','009','01','011']:
# 	cnt_sg('../data/isg3/df_ptsg_%s.csv'%s)

# for i in range(5):
# 	print 'fold%d'%i
# 	for s in ['001','002','006']:
# 		print s
# 		print cnt_sg_contains_0('../data/test/last12h_mean_mimic_m1_s0.%s_fold%d.out'%(s,i))
