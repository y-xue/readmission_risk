"""
Generate tensor file from csv files of MIMIC data per Rohit's output.
yluo - 04/29/2013 creation

"""

#! /usr/bin/python;
import csv
import re
import os
import glob
import shutil
import sys
import math
import mimic_stats as ms

sg_ub = 15000
tcmax = 6
def cnt_hash(k, h, hstop={}):
    if hstop.has_key(k):
        return None
    if not h.has_key(k) :
        h[k] = len(h)+1
    return h[k]

def interesting_sg(fnsg, nnodes_cut=tcmax, isg=0):
    """
    need subgraphs that doesn't contain NaN, isn't all 0, start 0, 
    end multi0 and has more than 1 nodes
    """
    hisg = {}
    fsg = open(fnsg, 'r')
    hn = {}; he = {}
    for ln in fsg:
        ln = ln.rstrip(" \n")
        # mn = re.search(r'^n (\d+) (\S+_n|\S+_m|\S+_p|loc)_(\S+)$', ln)
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
            hasNaN = "nan" in set(hn.values())
            all0 = set(['0']) == set(hn.values())
            startMulti0 = re.search("^0 0", sgstr)
            endMulti0 = re.search("0 0$", sgstr)
            start0 = re.search("^0", sgstr)
            end0 = re.search("0$", sgstr)
            locUnspecified = test=="loc" and ("0" in set(hn.values()) or "1" in set(hn.values()))
            if isg == 0:
                if (sup < sg_ub and not hasNaN and not all0 and not startMulti0 and not endMulti0 and nnodes > 1): # not locUnspecified and not (start0 and end0)
                    hisg[gid] = "(%s)\t%s\t%s\t[%s]" % (sup, test, sgstr, nnodes)
            elif isg == 1:
                # include single node graphs
                if (sup < sg_ub and not hasNaN and (nnodes == 1 or (not all0 and not startMulti0 and not endMulti0 and nnodes > 1))):
                    hisg[gid] = "(%s)\t%s\t%s\t[%s]" % (sup, test, sgstr, nnodes)
            elif isg == 2:
                # same as isg1 but no subiso
                # it's a coding trick, should be removed later
                if (sup < sg_ub and not hasNaN and (nnodes == 1 or (not all0 and not startMulti0 and not endMulti0 and nnodes > 1))):
                    hisg[gid] = "(%s)\t%s\t%s\t[%s]" % (sup, test, sgstr, nnodes)
            elif isg == 3:
                if (sup < sg_ub and not hasNaN and (endMulti0 or startMulti0) and nnodes > 1): # not locUnspecified and not (start0 and end0)
                    hisg[gid] = "(%s)\t%s\t%s\t[%s]" % (sup, test, sgstr, nnodes)
            elif isg == 4:
                # same as isg0 but no subiso
                # it's a coding trick, should be removed later
                if (sup < sg_ub and not hasNaN and not all0 and not startMulti0 and not endMulti0 and nnodes > 1): # not locUnspecified and not (start0 and end0)
                    hisg[gid] = "(%s)\t%s\t%s\t[%s]" % (sup, test, sgstr, nnodes)
            elif isg == 5:
                # used for checking if we can get the same result again
                # same as isg0
                # but NMF random-state = 2222
                if (sup < sg_ub and not hasNaN and not all0 and not startMulti0 and not endMulti0 and nnodes > 1): # not locUnspecified and not (start0 and end0)
                    hisg[gid] = "(%s)\t%s\t%s\t[%s]" % (sup, test, sgstr, nnodes)
            hn = {}; he = {}

    fsg.close()
    return hisg;

def read_graph(fngraph, fnnp):
    "The current configuration is to only read the property."
    hgs = {}; hg = {}
    # read the original graph
    fgraph = open(fngraph, "r")
    for ln in fgraph:
        # mv (match 'v')
        mv = re.search(r'^v (\d+) (.*)$', ln)
        if mv:
            # lab stands for the value of measures
            nid = int(mv.group(1)); nlab = mv.group(2)
            hg[nid] = {}
            hg[nid]['name'] = nlab
            hg[nid]['property'] = []
        # me (match 'e')
        me = re.search(r'^e (\d+) (\d+) (.*)$', ln)
        if me:
            n1 = int(me.group(1)); n2 = int(me.group(2)); elab = me.group(3)
        mg = re.search(r'^g ([\d_]+)$', ln)
        if mg:
            gid = mg.group(1)
            hgs[gid] = hg; hg = {}
    fgraph.close()
    
    # read additional property list
    fnp = open(fnnp, "r")
    hns = {}; nid = -1
    for ln in fnp:
        mv = re.search(r'^v (\d+)', ln)
        if mv:
            nid = int(mv.group(1))
        mp = re.search(r'\[(.*)\]', ln)
        if mp:
            np = mp.group(1); hns[nid] = np
        mg = re.search(r'g ([\d_]+)', ln)
        if mg:
            gid = mg.group(1)
            for nid in hns.keys():
                hgs[gid][nid]['property'] += hns[nid].rstrip(" ").split()
            hns = {}
    return hgs;


def append_tensor(ht, ft):
    """ dump and clear the current tenosr buffer """
    for k in ht.keys():
        ft.write("%s %.2f\n" % (k, ht[k] )) # math.log(ht[k]+1, 2)
    ht.clear()
    return;

def append_matrix(hm, fm):
    """ dump and clear the current matrix buffer """
    for k in hm.keys():
        fm.write("%s %.2f\n" % (k, hm[k] )) # math.log(hm[k]+1, 2)
    hm.clear()
    return;

def write_cnt_index(h, fn):
    ks = sorted(h.keys(), key=lambda k: h[k])
    f = open(fn, "w")
    for k in ks:
        f.write("%s\n" % (k))
    f.close()
    return;

def write_sg_str(hsgtid, hsg, fn):
    ks = sorted(hsgtid.keys(), key=lambda k: hsgtid[k])
    f = open(fn, "w")
    for k in ks:
        f.write("%s %s\n" % (k, hsg[k]))
    f.close()
    return;

def tensor_inc(htensor, t1, t2, t3, inc):
    """ tensor increment """
    if t1 == None or t2 == None or t3 == None:
        return
    tkey = "%s %s %s" % (t1, t2, t3)
    if htensor.has_key(tkey):
        htensor[tkey] += inc
    else:
        htensor[tkey] = inc
    return;

def matrix_inc(hmat, m1, m2, inc):
    """ matrix increment """
    if m1 == None or m2 == None:
        return
    mkey = "%s %s" % (m1, m2)
    if hmat.has_key(mkey):
        hmat[mkey] += inc
    else:
        hmat[mkey] = inc
    return;

def write_pt_mortc(hgtid, fnptmc, hmort):
    fptmc = open(fnptmc, "w")
    for ptid in hgtid.keys():
        fptmc.write("%s 1 %s\n" % (hgtid[ptid], hmort[int(ptid)]))
    fptmc.close();
    
def write_pt_w_matrix(hpttid, hntid, hg, fnmpt_w, hstop = {}):
    hmpt_w = {}
    fmpt_w = open(fnmpt_w, "w")
    for gid in hg.keys():
        m = re.search(r'(\d+)_(\d+)', gid)
        ptid = gid
        if m:
            ptid = m.group(1)
        ptid = int(ptid)
        if hpttid.has_key(ptid):
            pttid = hpttid[ptid]
            for gnid in hg[gid].keys():
                nstr = hg[gid][gnid]['name']
                if not re.search("n_(0|nan)", nstr): # |loc_(0|1)
                    ntid = cnt_hash(nstr, hntid, hstop)
                    matrix_inc(hmpt_w, pttid, ntid, 1)
        else:
            print("missing %s in pt by w" % (ptid))
    append_matrix(hmpt_w, fmpt_w)
    fmpt_w.close();

def pt_sg_w_tensor_gen(fngraph, fnnp, fnsg, fnmap, fntensor, fnmpt_sg, fnmpt_w, fnpttid, fnsgtid, fnsgstr, fnntid, fnmstop="../data/measure_stopwords", isg=0):
    """ Generate the tensor for the mimic data
    Input:
    fngraph (I) - graph file name (patient)
    fnnp (I) - additional properties file for nodes 
    fnmap (I) - file name of the mapping between graph and subgraph
    fntensor (O) - tensor file name
    fnpttid (O) - file name of graph index for tensor
    fnsgtid (O) - file name of subgraph index for tensor
    fnntid (O) - file name of node label index for tensor
    """
# mt.pt_sg_w_tensor_gen("data/mimic.nel", "data/mimic.node",   (fngraph,fnnp)
    # "data/mimic_m1_s0.001.out", "data/mimic_m1_s0.001.ids",  (fnsg, fnmap)
    # "data/pt_sg_w/mimic.tensor", "data/pt_sg_w/mimic_pt_sg.spmat",  (fntensor,fnmpt_sg)
    # "data/pt_sg_w/mimic_pt_w.spmat", "data/pt_sg_w/mimic.ptid",  (fnmpt_w, fnpttid)
    # "data/pt_sg_w/mimic.sgtid", "data/pt_sg_w/mimic.sgstr",  (fnsgtid, fnsgstr)
    # "data/pt_sg_w/mimic.ntid") (fnntid)
    # print 'tensor_gen isg:', isg
    # PaTient Time InDex
    hpttid = {};

    # SubGraph Time InDex
    hsgtid = {};

    # ?
    hntid = {};

    # tensor
    htensor = {}; 

    # matrix of PaTients & SubGraphs
    # pt = 15952, sg = 15952_70
    # pttid = 3, sgtid = 3, cnt = 2
    # m['3 3'] = 2
    hmpt_sg = {};

    hmpt_w = {}
    hg = read_graph(fngraph, fnnp) # '20389_38:{1: {'property': [], 'name': 'temp_n_0'}'
    hisg = interesting_sg(fnsg,isg=isg) # 5602 pts after interesting_sg()
    hmort = ms.read_mortality_rohit("../data/alldata_readmit.csv")
    h_meas_stop = {}; # load_stopwords(fnmstop)
    fmap = open(fnmap, "r")
    sgtid = 0; skip_sg = 0
    ftensor = open(fntensor, "w")
    fmpt_sg = open(fnmpt_sg, "w")

    hmatch = {}; current_pair = ""
    # match graph count?
    hmgcnt = {};

    # time entry??
    # htk="%s %s %s" % (pttid, sgtid, ntid) 
    # e.g. htk = "3 3 10", cnt = 6
    # ht_entry = {"3 3 10": 6}
    ht_entry = {}

    for ln in fmap:
        m = re.search(r'^(\d+):(.*)$', ln) # match 1:15952_70
        if m:
            sgid = int(m.group(1)) # m=[1, 15952_70], sgid = 1
            # skip interesting subgraph
            if hisg.has_key(sgid):
                skip_sg = 0
            else:
                skip_sg = 1
                continue
            sgtid = cnt_hash(sgid, hsgtid)  # hsgtid:{2:1,5:2,1:3}
                                            # hsgtid:{sgid:sgtid}
                                            # means (sgid==2) is visited first, (sgid == 5) is vistied secondly
                                            #
                                            # sgtid indicates the time (or the order)
                                            # that sgid is visited

        m = re.search(r'^([\d_]+) -> (.*) $', ln) # match: 15952_70 -> 2 1 3 4 5 6 
        if m and not skip_sg:
            gid = m.group(1) # gid = 15952_70
            pair = "%s %s" % (sgid, gid) # (1,15952_70)
            if current_pair != pair:
                hmatch.clear()
                current_pair = pair
            gns = m.group(2) # gns = '2 1 3 4 5 6'
            gns = gns.split()
            gnset = "%s" % (sorted(gns))
            if hmatch.has_key(gnset):
                continue # if it's simply permutation, do nothing
                            # simple permutation perhaps means '1 2 3 4 5 6'
            else:
                hmatch[gnset] = 1

            mgid = re.search(r'(\d+)_(\d+)', gid)
            ptid = gid
            if mgid:
                ptid = int(mgid.group(1)) # PaTientID = 15952
            else:
                ptid = int(gid)
            pttid = cnt_hash(ptid, hpttid)  # pttid indicates the time (or the order)
                                            # that ptid is visited

            # hmgcnt = {pttid: {sgtid: cnt}}
            # for (1,15952_70), which is visited at 3nd place
            # hmgcnt = {3:{3: 1}}
            if not hmgcnt.has_key(pttid):
                hmgcnt[pttid] = {}
            if not hmgcnt[pttid].has_key(sgtid):
                hmgcnt[pttid][sgtid] = 0
            hmgcnt[pttid][sgtid] += 1

            matrix_inc(hmpt_sg, pttid, sgtid, 1)
            
            # ['1', '2', '3', '4', '5', '6']
            for sgnid in gns:
                gnid = int(sgnid)
                for nd_wt in hg[gid][gnid]['property']:
                    [nstr, wt] = nd_wt.split(":") # e.g. nd_wt='RBC_n_-2:-1.487073'
                    # if nstr in ['loc_0', 'loc_1']:
                    #     continue 
                    ntid = cnt_hash(nstr, hntid, h_meas_stop)   # ntid indicates the time (or the order)
                                                                # that nstr is visited
                    htk = "%s %s %s" % (pttid, sgtid, ntid) # e.g. htk = "3 3 10"
                    if ht_entry.has_key(htk):
                        ht_entry[htk] += 1. 
                    else:
                        ht_entry[htk] = 1. 

    for htk in ht_entry.keys():
        [pttid, sgtid, ntid] = map(int, htk.split())
        htv = ht_entry[htk] # cnt of htk
        tensor_inc(htensor, pttid, sgtid, ntid, htv/hmgcnt[pttid][sgtid])   # how many pt_sg_n / how many pt_sg
                                                                            # n stands for measures (like mMAP_n_1)
                                                                            #
                                                                            # htv/hmgcnt[][] means frequency of n
                                                                            # in certain pt_sg

    append_tensor(htensor, ftensor)
    append_matrix(hmpt_sg, fmpt_sg)
    
    write_pt_w_matrix(hpttid, hntid, hg, fnmpt_w)
    fmap.close()
    ftensor.close()
    fmpt_sg.close()

    # output index files
    write_cnt_index(hpttid, fnpttid)
    write_cnt_index(hsgtid, fnsgtid)
    write_sg_str(hsgtid, hisg, fnsgstr)
    write_cnt_index(hntid, fnntid)
    fnptmc = fnpttid.replace(".ptid", ".ptmc")
    write_pt_mortc(hpttid, fnptmc, hmort)
    return;
