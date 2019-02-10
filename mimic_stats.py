"""
Generate graph database from csv files of MIMIC data per Rohit's output.
Modified by yxue - 05-18-2017
First created by yluo - 04/28/2013 creation

"""

#! /usr/bin/python;
import csv
import re
import os
import glob
import shutil
# import psycopg2 as pg
import numpy as np
# from scipy import stats
from datetime import datetime

def read_mortality_rohit(fn):
    hmc = {}
    f = open(fn, "r")
    freader = csv.reader(f, delimiter=',', quotechar="\"")
    lcnt = 0
    for row in freader:
        lcnt += 1
        if lcnt > 1:
            sid = int(row[1]); mc = int(row[0])+1
            hmc[sid] = mc
    f.close()
    hist = {}
    for k in hmc.keys():
        v = hmc[k]
        if hist.has_key(v):
            hist[v] += 1
        else:
            hist[v] = 1
    for k in hist.keys():
        print("%s: %s" % (k, hist[k]))
              
    return hmc

def read_icd9(fnicd9, fngraph):
    hicd9 = {}; hcnt = {}; hpt = {}
    fgraph = open(fngraph, "r")
    for ln in fgraph:
        ln = ln.rstrip(" \n")
        m = re.search(r'^g (\d+)_', ln)
        if m:
            ptid = int(m.group(1))
            hpt[ptid] = 1
    fgraph.close()


    ficd9 = open(fnicd9, "r")
    freader = csv.reader(ficd9, delimiter=',', quotechar="\"", quoting=csv.QUOTE_MINIMAL)
    for row in freader:
        (sid, icd9) = row
        icd9 = icd9.rstrip(" \n")
        if hpt.has_key(int(sid)):
            for code in icd9.split():
                if hcnt.has_key(code):
                    hcnt[code] += 1
                else:
                    hcnt[code] = 1
    ficd9.close()

    ficd9 = open(fnicd9, "r")
    freader = csv.reader(ficd9, delimiter=',', quotechar="\"", quoting=csv.QUOTE_MINIMAL)
    for row in freader:
        (sid, icd9) = row
        sid = int(sid); icd9 = icd9.rstrip(" \n")
        hicd9[sid] = {}
        for code in icd9.split():
            if hpt.has_key(sid) and hcnt[code] >= 2:
                # code = re.sub(r'\..*$', "", code)
                hicd9[sid][code] = 1
    ficd9.close()
    return hicd9;

def read_pt_info(fnpti):
    hgender = {}; hage = {}; hcomorb = {}; hmort = {}
    fpti = open(fnpti, "r")
    freader = csv.reader(fpti, delimiter=',', quotechar="\"")
    lcnt = 0
    for row in freader:
        lcnt += 1
        if lcnt > 1:
            (sid, sex, daysfromfinaldischargetodeath, icustay_admit_age, icd9, congestive_heart_failure, cardiac_arrhythmias, valvular_disease, aids, alcohol_abuse, blood_loss_anemia, chronic_pulmonary, coagulopathy, deficiency_anemias, depression, diabetes_complicated, diabetes_uncomplicated, drug_abuse, fluid_electrolyte, hypertension, hypothyroidism, liver_disease, lymphoma, metastatic_cancer, obesity, other_neurological, paralysis, peptic_ulcer, peripheral_vascular, psychoses, pulmonary_circulation, renal_failure, rheumatoid_arthritis, solid_tumor, weight_loss) = row
            sid = int(sid); age = float(icustay_admit_age)
            days = int(daysfromfinaldischargetodeath)
            hgender[sid] = sex; hage[sid] = discretize_age(age)
            mc = discretize_mort(days)
            if mc != -1:
                hmort[sid] = mc
            icd9 = icd9.rstrip(" \n")
            hcomorb[sid] = {}
            for code in icd9.split():
                hcomorb[sid][code] = 1
            # hcomorb[sid]['congestive_heart_failure'] = int(congestive_heart_failure)
            # hcomorb[sid]['cardiac_arrhythmias'] = int(cardiac_arrhythmias)
            # hcomorb[sid]['valvular_disease'] = int(valvular_disease)
            # hcomorb[sid]['aids'] = int(aids)
            # hcomorb[sid]['alcohol_abuse'] = int(alcohol_abuse)
            # hcomorb[sid]['blood_loss_anemia'] = int(blood_loss_anemia)
            # hcomorb[sid]['chronic_pulmonary'] = int(chronic_pulmonary)
            # hcomorb[sid]['coagulopathy'] = int(coagulopathy)
            # hcomorb[sid]['deficiency_anemias'] = int(deficiency_anemias)
            # hcomorb[sid]['depression'] = int(depression)
            # hcomorb[sid]['diabetes_complicated'] = int(diabetes_complicated)
            # hcomorb[sid]['diabetes_uncomplicated'] = int(diabetes_uncomplicated)
            # hcomorb[sid]['drug_abuse'] = int(drug_abuse)
            # hcomorb[sid]['fluid_electrolyte'] = int(fluid_electrolyte)
            # hcomorb[sid]['hypertension'] = int(hypertension)
            # hcomorb[sid]['hypothyroidism'] = int(hypothyroidism)
            # hcomorb[sid]['liver_disease'] = int(liver_disease)
            # hcomorb[sid]['lymphoma'] = int(lymphoma)
            # hcomorb[sid]['metastatic_cancer'] = int(metastatic_cancer)
            # hcomorb[sid]['obesity'] = int(obesity)
            # hcomorb[sid]['other_neurological'] = int(other_neurological)
            # hcomorb[sid]['paralysis'] = int(paralysis)
            # hcomorb[sid]['peptic_ulcer'] = int(peptic_ulcer)
            # hcomorb[sid]['peripheral_vascular'] = int(peripheral_vascular)
            # hcomorb[sid]['psychoses'] = int(psychoses)
            # hcomorb[sid]['pulmonary_circulation'] = int(pulmonary_circulation)
            # hcomorb[sid]['renal_failure'] = int(renal_failure)
            # hcomorb[sid]['rheumatoid_arthritis'] = int(rheumatoid_arthritis)
            # hcomorb[sid]['solid_tumor'] = int(solid_tumor)
            # hcomorb[sid]['weight_loss'] = int(weight_loss)
    fpti.close()        
    return (hmort, hgender, hage, hcomorb)