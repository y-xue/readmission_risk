"""
Generate graph database from csv files of MIMIC data per Rohit's output, using 
interpolation.
Modified by yxue - 05-18-2017
First created by yluo - 05/18/2013
"""

#! /usr/bin/python;
import csv
import re
import os
import glob
import shutil
import sys
import mimic_stats as ms
import nel_graph_gen as ngg
from copy import deepcopy
from numpy import *

class NelGraphGenInterpolation(ngg.NelGraphGen):
    tcmax = 6; 
    tu = 60*2; # 2 hrs intervals, 12 hrs
    toffset = 720; # toffset changes will affect has_last_12h

    def __init__(self):
        self.hnancnt = {}
        self.hirreg = {}
    
    def add_time(self, ncnt, time, prev_time, prev_tidx, hlab, prev_hlab, raw=False):
        "For now ignore the actual time"
        etext = {}; ncnt += 1; time = int(time); prev_time = int(prev_time)
        ntext = {}
        for k in hlab.keys():
            lab = hlab[k]
            if k == "olab":
                continue;
                lab = super(NelGraphGenInterpolation, self).collapse_olab(lab)
            elif k == "mlab":
                continue 
                lab = int(round(lab))
            # elif k in ['tProtein_n', 'ASTtoALT_n', 'AST_n', 'ALT_n', 'TBili_n', 'DBili_n', 'Albumin_n', 'Lactate_n']:
            #     continue
            elif re.search(r'_n$', k):
                lab = super(NelGraphGenInterpolation, self).d_measure(lab, raw=False)
            elif re.search(r'_r$', k):
                lab = super(NelGraphGenInterpolation, self).d_measure(lab, raw=True)
            elif re.search(r'_l$', k):
                continue
                lab = super(NelGraphGenInterpolation, self).d_organlab(lab)
            elif re.search(r'_d$', k):
                continue
                if not isnan(lab):
                    lab = int(round(lab))
            elif k == "loc":
                lab = int(round(lab))
                if lab == 0:
                    lab = 1
            elif re.search(r'_(m|p)$', k): # meds and malig
                if not isnan(lab):
                    lab = int(lab)
            else: # others
                continue
            ntext[k] = "v %s %s_%s\n" % (ncnt, k, lab) 

            # mk = re.sub(r'_d$', "_m", k)
            # mlab = hlab[mk]
            # if not isnan(mlab):
            #     mlab = int(round(mlab))
            # ntext[k] += "_%s_%s\n" % (mk, mlab) # 

            if prev_tidx != -1: 
                # etext is the same for all organs
                plab = prev_hlab[k]
                if k == "olab":
                    plab = super(NelGraphGenInterpolation, self).collapse_olab(plab)
                elif k == "mlab":
                    plab = int(round(plab))
                elif re.search(r'_n$', k):
                    plab = super(NelGraphGenInterpolation, self).d_measure(plab, raw=False)
                elif re.search(r'_r$', k):
                    plab = super(NelGraphGenInterpolation, self).d_measure(plab, raw=True)
                elif re.search(r'_l$', k):
                    plab = super(NelGraphGenInterpolation, self).d_organlab(plab)
                elif re.search(r'_d$', k):
                    if not isnan(plab):
                        plab = int(round(plab))
                elif k == "loc":
                    plab = int(round(plab))
                    if plab == 0:
                        plab = 1
                elif re.search(r'_(m|p)$', k): # meds and malig
                    if not isnan(plab):
                        plab = int(plab)
                else: # others
                    continue

                if (plab == lab):
                    etext[k] = "e %s %s tsame\n" % (prev_tidx, ncnt) #, dtime
                elif plab < lab:
                    etext[k] = "e %s %s tup\n" % (prev_tidx, ncnt) #, dtime
                elif plab > lab:
                    etext[k] = "e %s %s tdown\n" % (prev_tidx, ncnt) # , dtime
                else:
                    etext[k] = "e %s %s t\n" % (prev_tidx, ncnt) 
            else:
                etext[k] = ""

        return (ntext, etext, ncnt);

    def has_1st_day(self, tidx):
        # print(tidx)
        # if len(tidx) > 1 and tidx[1] <= self.tu and tidx[len(tidx)-1] >= self.tu * self.tcmax:
        if len(tidx) > 3 and tidx[0] <= self.toffset + self.tu and tidx[-1] >= self.toffset + self.tu * self.tcmax: #  and tidx[-1] <= 1440 * 6:
            return True
        else:
            return False

    def has_last_day(self, tidx):
        # if len(tidx) > 1 and tidx[len(tidx)-1] - tidx[1] >= self.tu * self.tcmax and tidx[len(tidx)-1] <= 1440 * 6:
        if len(tidx) > 3:
            return True
        else:
            return False

    def has_two_days(self, tidx):
        if len(tidx) > 1 and tidx[1] <= self.tu and tidx[len(tidx)-1] >= self.tu * self.tcmax * 2:
            return True
        else:
            return False

    def has_last_12h(self, tidx):
        if len(tidx) > 1 and tidx[-1] >= self.toffset:
            return True
        else:
            return False

    def attach_node(self, hn, ncnt, tidx, time, elab):
        ntext = {}; etext = {}; ncnt += 1; time = int(time)
        for k in hn.keys():
            ntext[k] = "v %s %s_%s_%s\n" % (ncnt, k, hn[k], time)
            etext[k] = "e %s %s %s\n" % (tidx, ncnt, elab)
        return (ntext, etext, ncnt);

    def repeat_last(self, tidx_new, tidx, vals):
        vals_new = zeros(tidx_new.shape)
        j = 0
        for i in range(len(tidx_new)):
            while j < len(tidx) and tidx[j] <= tidx_new[i]:
                vals_new[i] = vals[j]; j += 1
        return vals_new;
    
    # def interpolating_raw(self, ptarr, sid, hmeans, hstds):
    #     tgraph = {}; tnode = {}; ntext = {}; etext = {}; hlab = {}
    #     (rows, cols) = ptarr.shape
    #     tidx = ptarr[:,0]; 
        
    #     # if not self.has_1st_day(tidx): # , has_last_day
    #     #     return (tgraph, tnode)
    #     # thridx = self.tu * (arange(self.tcmax) + 1)
    #     if not self.has_last_12h(tidx):
    #         return (tgraph, tnode)
    #     thridx = tidx[-1] - 720 + self.tu * (arange(self.tcmax) + 1)

    #     # the line below offset to the last day
    #     # thridx = tidx[len(tidx)-1] - thridx[len(thridx)-1] + thridx

        
    #     thridx = thridx.reshape(self.tcmax, 1); iptarr = thridx
    #     for ci in range(1,cols-2):
    #         y = ptarr[:,ci]
    #         yi = interp(thridx, tidx, y)
    #         iptarr = hstack((iptarr, yi))
    #     # med_l
    #     y = ptarr[:,cols-2]
    #     yi = self.repeat_last(thridx, tidx, y)
    #     iptarr = hstack((iptarr, yi))
    #     # loc
    #     y = ptarr[:,cols-1]
    #     yi = self.repeat_last(thridx, tidx, y)
    #     iptarr = hstack((iptarr, yi))


    #     ncnt = 0; prev_tidx = -1; prev_time = 0; prev_hlab = {}
    #     for hr in range(self.tcmax):
    #         [time,
    #          # hlab['olab'], hlab['kidney_l'], hlab['lvr_l'], hlab['hemat_l'], hlab['lung_l'], hlab['cardio_l'], hlab['acidbase_l'], hlab['elytes_l'], hlab['general_l'], hlab['kidney_m'], hlab['lvr_m'], hlab['hemat_m'], hlab['lung_m'], hlab['cardio_m'], hlab['acidbase_m'], hlab['elytes_m'], hlab['general_m'], hlab['kidney_d'], hlab['lvr_d'], hlab['hemat_d'], hlab['lung_d'], hlab['cardio_d'], hlab['acidbase_d'], hlab['elytes_d'], hlab['general_d'],
    #          # meds
    #          # hlab['m_Antiarrhythmic'], hlab['m_Anticoagulant'], hlab['m_Antiplatelet'], hlab['m_Benzodiazepine'], hlab['m_beta_Blocking'], hlab['m_Calcium_channel_blocking'], hlab['m_Diuretic'], hlab['m_Hemostatic'], hlab['m_Inotropic'], hlab['m_Insulin'], hlab['m_Nondepolarizing'], hlab['m_sedatives'], hlab['m_Somatostatin_preparation'], hlab['m_Sympathomimetic'], hlab['m_Thrombolytic'], hlab['m_Vasodilating'],
    #          # 
    #          hlab['Creatinine'], hlab['BUN'], hlab['BUNtoCr'], hlab['urineByHrByWeight'], hlab['eGFR'], hlab['AST'], hlab['ALT'], hlab['TBili'], hlab['DBili'], hlab['Albumin'], hlab['tProtein'], hlab['ASTtoALT'], hlab['HCT'], hlab['Hgb'], hlab['INR'], hlab['Platelets'], hlab['PT'], hlab['PTT'], hlab['RBC'], hlab['WBC'], hlab['RESP'], hlab['mSaO2'], hlab['PaO2toFiO2'], hlab['MinuteVent'], hlab['DeliveredTidalVolume'], hlab['FiO2Set'], hlab['PEEPSet'], hlab['PIP'], hlab['RSBI'], hlab['RSBIRate'], hlab['RAW'], hlab['PlateauPres'], hlab['mSBP'], hlab['mDBP'], hlab['mMAP'], hlab['CV_HR'], hlab['mCrdIndx'], hlab['mCVP'], hlab['Art_BE'], hlab['Art_CO2'], hlab['Art_PaCO2'], hlab['Art_PaO2'], hlab['Art_pH'], hlab['Na'], hlab['K'], hlab['Cl'], hlab['Glucose'], hlab['Ca'], hlab['Mg'], hlab['IonCa'], hlab['Lactate'], hlab['GCS'], hlab['temp'],
    #          # 
    #          hlab['Creatinine_n'], hlab['BUN_n'], hlab['BUNtoCr_n'], hlab['urineByHrByWeight_n'], hlab['eGFR_n'], hlab['AST_n'], hlab['ALT_n'], hlab['TBili_n'], hlab['DBili_n'], hlab['Albumin_n'], hlab['tProtein_n'], hlab['ASTtoALT_n'], hlab['HCT_n'], hlab['Hgb_n'], hlab['INR_n'], hlab['Platelets_n'], hlab['PT_n'], hlab['PTT_n'], hlab['RBC_n'], hlab['WBC_n'], hlab['RESP_n'], hlab['mSaO2_n'], hlab['PaO2toFiO2_n'], hlab['MinuteVent_n'], hlab['DeliveredTidalVolume_n'], hlab['FiO2Set_n'], hlab['PEEPSet_n'], hlab['PIP_n'], hlab['RSBI_n'], hlab['RSBIRate_n'], hlab['RAW_n'], hlab['PlateauPres_n'], hlab['mSBP_n'], hlab['mDBP_n'], hlab['mMAP_n'], hlab['CV_HR_n'], hlab['mCrdIndx_n'], hlab['mCVP_n'], hlab['Art_BE_n'], hlab['Art_CO2_n'], hlab['Art_PaCO2_n'], hlab['Art_PaO2_n'], hlab['Art_pH_n'], hlab['Na_n'], hlab['K_n'], hlab['Cl_n'], hlab['Glucose_n'], hlab['Ca_n'], hlab['Mg_n'], hlab['IonCa_n'], hlab['Lactate_n'], hlab['GCS_n'], hlab['temp_n'], hlab['mlab'], hlab['loc']] = iptarr[hr,:]
    #         hp = {}
    #         for k in hlab.keys():
    #             if isnan(hlab[k]):
    #                 if not self.hnancnt.has_key(k):
    #                     self.hnancnt[k] = 0
    #                 self.hnancnt[k] += 1

    #             if re.search(r'_n$', k) and not isnan(hlab[k]):
    #                 korig = re.sub(r'_n$', "", k)
    #                 if korig == "PaO2toFiO2":
    #                     if isnan(hlab["Art_PaO2"]) or isnan(hlab["FiO2Set"]):
    #                         hlab[k] = nan; hlab[korig] = nan
    #                 if korig == "ASTtoALT":
    #                     if isnan(hlab["AST"]) or isnan(hlab["ALT"]):
    #                         hlab[k] = nan; hlab[korig] = nan
    #                 if korig == "BUNtoCr":
    #                     if isnan(hlab["BUN"]) or isnan(hlab["Creatinine"]):
    #                         hlab[k] = nan; hlab[korig] = nan  
    #                 if korig == "eGFR":
    #                     if isnan(hlab["Creatinine"]):
    #                         hlab[k] = nan; hlab[korig] = nan  
    #                 if korig == "DeliveredTidalVolume":
    #                     if isnan(hlab["PIP"]): # or isnan(hlab["TidVolObs"])
    #                         hlab[k] = nan; hlab[korig] = nan 
    #                 if korig == "MinuteVent":
    #                     if isnan(hlab["RESP"]): # or isnan(hlab["TidVolObs"])
    #                         hlab[k] = nan; hlab[korig] = nan 
    #                 if korig == "RSBI":
    #                     if isnan(hlab["RESP"]) or isnan(hlab["DeliveredTidalVolume"]):
    #                         hlab[k] = nan; hlab[korig] = nan 
    #                 if korig == "RAW":
    #                     if isnan(hlab["PIP"]) or isnan(hlab["PlateauPres"]):
    #                         hlab[k] = nan; hlab[korig] = nan 

    #                 if isnan(hlab[korig]):
    #                     hlab[k] = hlab[korig]
    #                 else:
    #                     hlab[k] = (hlab[korig] - hmeans[korig]) / hstds[korig]
    #                     hp[k] = super(NelGraphGenInterpolation, self).d_measure(hlab[k], raw=True)
                    

    #                 if isnan(hlab[korig]):
    #                     if not self.hirreg.has_key(sid):
    #                         self.hirreg[sid] = {}
    #                     if not self.hirreg[sid].has_key(korig):
    #                         self.hirreg[sid][korig] = 1
    #                         print('nan irreg %s:%s' % (sid, korig))

    #             elif re.search(r'^m_', k) and hlab[k] == 1:
    #                 hp[k] = int(hlab[k])
                
    #             elif re.search(r'_d$', k) and not isnan(hlab[k]):
    #                 continue
    #                 hp[k] = int(round(hlab[k]))
    #             elif re.search(r'_m$', k) and not isnan(hlab[k]):
    #                 continue
    #                 hp[k] = int(round(hlab[k]))
    #             elif k == "loc":
    #                 hp[k] = int(round(hlab[k]))
    #                 if hp[k] == 0:
    #                     hp[k] = 1

    #         (t_ntext, t_etext, ncnt) = self.add_time(ncnt, time, prev_time, prev_tidx, hlab, prev_hlab, raw=True)
    #         tidx = ncnt; 
    #         for k in t_ntext.keys():
    #             if ntext.has_key(k):
    #                 ntext[k] += t_ntext[k]
    #                 etext[k] += t_etext[k]
    #             else:
    #                 ntext[k] = t_ntext[k]
    #                 etext[k] = t_etext[k]
    #             if tnode.has_key(k):
    #                 tnode[k] += t_ntext[k]
    #             else:
    #                 tnode[k] = t_ntext[k]
    #             tnode[k] += ("[%s]\n" % super(NelGraphGenInterpolation, self).attach_property(hp, hlab, time))

    #         prev_time = time; prev_tidx = tidx; prev_hlab = deepcopy(hlab)
        
    #     for k in ntext.keys():
    #         tgraph[k] = "%s%s" % (ntext[k], etext[k]); 
    #     return (tgraph, tnode);             

    def interpolating(self, ptarr, sid, cz=False):
        tgraph = {}; tnode = {}; ntext = {}; etext = {}; hlab = {}
        (rows, cols) = ptarr.shape
        tidx = ptarr[:,0]; # list of timeindex
        
        # if not self.has_1st_day(tidx): #  , has_last_day
        #     return (tgraph, tnode)
        if not self.has_last_12h(tidx):
            return (tgraph, tnode)
        thridx = tidx[-1] - 720 + self.tu * (arange(self.tcmax) + 1)

        # thridx = self.toffset + self.tu * (arange(self.tcmax) + 1)
        # the line below offset to the last day
        # thridx = tidx[len(tidx)-1] - thridx[len(thridx)-1] + thridx
        
        thridx = thridx.reshape(self.tcmax, 1); iptarr = thridx
        # thridx = reshape([t1, t2, ..., tn], (n,1))

        # iptarr (patient array of certain times)
        # [time, 'Creatinine', 'BUN', ...] (features)
        # [t1, xxx, xxx, ...]
        # [t2, xxx, xxx, ...]
        # ...
        # [tn, xxx, xxx, ...]

        for ci in range(1,cols-2):
            y = ptarr[:,ci]                 # values of feature ci
            yi = interp(thridx, tidx, y)    # get interp for certain times in
                                            # mapping from time to feature value
            iptarr = hstack((iptarr, yi))
        # med_l, which doesn't make sense for linear interpolation
        y = ptarr[:,cols-2]
        yi = self.repeat_last(thridx, tidx, y)
        iptarr = hstack((iptarr, yi))
        # loc, which doesn't make sense for linear interpolation
        y = ptarr[:,cols-1]
        yi = self.repeat_last(thridx, tidx, y)
        iptarr = hstack((iptarr, yi))
        
        # only low
        # if max(iptarr[:,1]) <= 3:
        #     return (tgraph, tnode)

        ncnt = 0; prev_tidx = -1; prev_time = 0; prev_hlab = {}
        for hr in range(self.tcmax):
            if cz:
                [time,
                hlab['Creatinine_n'], hlab['BUN_n'], hlab['BUNtoCr_n'], hlab['urineByHrByWeight_n'], hlab['eGFR_n'], hlab['AST_n'], hlab['ALT_n'], hlab['TBili_n'], hlab['DBili_n'], hlab['Albumin_n'], hlab['tProtein_n'], hlab['ASTtoALT_n'], hlab['HCT_n'], hlab['Hgb_n'], hlab['INR_n'], hlab['Platelets_n'], hlab['PT_n'], hlab['PTT_n'], hlab['RBC_n'], hlab['WBC_n'], hlab['RESP_n'], hlab['mSaO2_n'], hlab['PaO2toFiO2_n'], hlab['MinuteVent_n'], hlab['DeliveredTidalVolume_n'], hlab['FiO2Set_n'], hlab['PEEPSet_n'], hlab['PIP_n'], hlab['RSBI_n'], hlab['RSBIRate_n'], hlab['RAW_n'], hlab['PlateauPres_n'], hlab['mSBP_n'], hlab['mDBP_n'], hlab['mMAP_n'], hlab['CV_HR_n'], hlab['mCrdIndx_n'], hlab['mCVP_n'], hlab['Art_BE_n'], hlab['Art_CO2_n'], hlab['Art_PaCO2_n'], hlab['Art_PaO2_n'], hlab['Art_pH_n'], hlab['Na_n'], hlab['K_n'], hlab['Cl_n'], hlab['Glucose_n'], hlab['Ca_n'], hlab['Mg_n'], hlab['IonCa_n'], hlab['Lactate_n'], hlab['GCS_n'], hlab['temp_n'],
                hlab['Antiarrhythmic_m'], hlab['Anticoagulant_m'], hlab['Antiplatelet_m'], hlab['Benzodiazepine_m'], hlab['beta.Blocking_m'], hlab['Calcium_channel_blocking_m'], hlab['Diuretic_m'], hlab['Hemostatic_m'], hlab['Inotropic_m'], hlab['Insulin_m'], hlab['Nondepolarizing_m'], hlab['sedatives_m'], hlab['Somatostatin_preparation_m'], hlab['Sympathomimetic_m'], hlab['Thrombolytic_m'], hlab['Vasodilating_m'],
                hlab['AIDS_p'], hlab['HemMalig_p'], hlab['MetCarcinoma_p'],
                hlab['mlab'], hlab['loc']] = iptarr[hr,:]
            else:
                [time,
                hlab['Creatinine_r'], hlab['BUN_r'], hlab['BUNtoCr_r'], hlab['urineByHrByWeight_r'], hlab['eGFR_r'], hlab['AST_r'], hlab['ALT_r'], hlab['TBili_r'], hlab['DBili_r'], hlab['Albumin_r'], hlab['tProtein_r'], hlab['ASTtoALT_r'], hlab['HCT_r'], hlab['Hgb_r'], hlab['INR_r'], hlab['Platelets_r'], hlab['PT_r'], hlab['PTT_r'], hlab['RBC_r'], hlab['WBC_r'], hlab['RESP_r'], hlab['mSaO2_r'], hlab['PaO2toFiO2_r'], hlab['MinuteVent_r'], hlab['DeliveredTidalVolume_r'], hlab['FiO2Set_r'], hlab['PEEPSet_r'], hlab['PIP_r'], hlab['RSBI_r'], hlab['RSBIRate_r'], hlab['RAW_r'], hlab['PlateauPres_r'], hlab['mSBP_r'], hlab['mDBP_r'], hlab['mMAP_r'], hlab['CV_HR_r'], hlab['mCrdIndx_r'], hlab['mCVP_r'], hlab['Art_BE_r'], hlab['Art_CO2_r'], hlab['Art_PaCO2_r'], hlab['Art_PaO2_r'], hlab['Art_pH_r'], hlab['Na_r'], hlab['K_r'], hlab['Cl_r'], hlab['Glucose_r'], hlab['Ca_r'], hlab['Mg_r'], hlab['IonCa_r'], hlab['Lactate_r'], hlab['GCS_r'], hlab['temp_r'], 
                hlab['Antiarrhythmic_m'], hlab['Anticoagulant_m'], hlab['Antiplatelet_m'], hlab['Benzodiazepine_m'], hlab['beta.Blocking_m'], hlab['Calcium_channel_blocking_m'], hlab['Diuretic_m'], hlab['Hemostatic_m'], hlab['Inotropic_m'], hlab['Insulin_m'], hlab['Nondepolarizing_m'], hlab['sedatives_m'], hlab['Somatostatin_preparation_m'], hlab['Sympathomimetic_m'], hlab['Thrombolytic_m'], hlab['Vasodilating_m'],
                hlab['AIDS_p'], hlab['HemMalig_p'], hlab['MetCarcinoma_p'],
                hlab['mlab'], hlab['loc']] = iptarr[hr,:]

            hp = {}
            if sid==21743:
                print('hr=%d' % (hr))
            for k in hlab.keys():
                # count nan values of feature k
                if isnan(hlab[k]): # if feature k's value is nan
                    if not self.hnancnt.has_key(k):
                        self.hnancnt[k] = 0
                    self.hnancnt[k] += 1
                # print('|%s, %s|' % (k, hlab[k]))
                
                # for '_n' features:
                if re.search(r'_n$', k) and not isnan(hlab[k]):
                    # ????????????????????????????????????????????????????????????????????????????????????????
                    # if value of feature 'korig_n' is not 0
                    # if hlab[k] != 0:
                    hp[k] = super(NelGraphGenInterpolation, self).d_measure(hlab[k])
                elif re.search(r'_r$', k) and not isnan(hlab[k]):
                    hp[k] = super(NelGraphGenInterpolation, self).d_measure(hlab[k], raw=True)
                elif re.search(r'^m_', k) and hlab[k] == 1:
                    hp[k] = int(hlab[k])
                
                elif re.search(r'_d$', k) and not isnan(hlab[k]):
                    continue
                    hp[k] = int(round(hlab[k]))
                elif re.search(r'_(m|p)$', k) and not isnan(hlab[k]) and hlab[k]!=0: # meds and malig
                    hp[k] = int(round(hlab[k]))
                elif k == "loc":
                    hp[k] = int(round(hlab[k]))
                    if hp[k] == 0: # yluo 0202
                        hp[k] = 1
                    
            (t_ntext, t_etext, ncnt) = self.add_time(ncnt, time, prev_time, prev_tidx, hlab, prev_hlab)
            tidx = ncnt; 
            for k in t_ntext.keys():
                if ntext.has_key(k):
                    ntext[k] += t_ntext[k]
                    etext[k] += t_etext[k]
                else:
                    ntext[k] = t_ntext[k]
                    etext[k] = t_etext[k]
                if tnode.has_key(k):
                    tnode[k] += t_ntext[k]
                else:
                    tnode[k] = t_ntext[k]
                    
                if re.search("_(m|n|p)_(0|nan)", t_ntext[k]):
                    tnode[k] += "[]\n"
                else:
                    tnode[k] += ("[%s]\n" % super(NelGraphGenInterpolation, self).attach_property(hp, hlab, time))

            # hd = {}
            # for k in hlab.keys():
            #     if re.search(r'_m$', k):
            #         if isnan(hlab[k]):
            #             hd[k] = hlab[k]
            #         else:
            #             hd[k] = int(round(hlab[k]))
            # (d_ntext, d_etext, ncnt) = self.attach_node(hd, ncnt, tidx, time, "mag")
            # for k in d_ntext.keys():
            #     mk = re.sub(r'_m$', "_d", k)
            #     ntext[mk] += d_ntext[k]
            #     etext[mk] += d_etext[k]


            prev_time = time; prev_tidx = tidx; prev_hlab = deepcopy(hlab)
        
        for k in ntext.keys():
            tgraph[k] = "%s%s" % (ntext[k], etext[k]); 
        return (tgraph, tnode);


    def write_graphs(self, fgraph, fnp, tgraph, tnode, sid):
        gcnt = 0
        for k in tgraph.keys():
            gcnt += 1
            fgraph.write("%sg %s_%s\n\n" % (tgraph[k], sid, gcnt))
            fnp.write("%sg %s_%s\n\n" % (tnode[k], sid, gcnt))
        return;

    def scan_csv_interpolation(self, fnin, fngraph, fnnp, hmeans={}, hstds= {}, partialPts=[], cz=False):
        "TODO: discretize timeindex"
        fin = open(fnin, 'r')
        
        freader = csv.reader(fin, delimiter=',', quotechar="\"")
        fgraph = open(fngraph, 'w')
        fnp = open(fnnp, "w")
        partialPts = map(int, partialPts)

        lcnt = 0; current_sid = ""; ptarr = None; gcnt = 0; current_mort = 0;
        # ptarr (patient array)
        # [time, 'Creatinine', 'BUN', ...]
        # [[0, xxx, xxx, ...],
        #  [1, xxx, xxx, ...],
        #  ...
        #  [last measure, xxx, xxx, ...]]
        hv = {}
        for row in freader:
            lcnt += 1
            if lcnt == 1:
                if cz:
                    vns = ['readmit', 'sid', 'timeindex',
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
                else:
                    vns = ['readmit', 'sid', 'timeindex',
                    'Creatinine_r', 'BUN_r', 'BUNtoCr_r', 'urineByHrByWeight_r', 'eGFR_r', 'AST_r', 'ALT_r', 'TBili_r', 'DBili_r',
                    'Albumin_r', 'tProtein_r', 'ASTtoALT_r', 'HCT_r', 'Hgb_r', 'INR_r',
                    'Platelets_r', 'PT_r', 'PTT_r', 'RBC_r', 'WBC_r', 'RESP_r', 'mSaO2_r', 'PaO2toFiO2_r', 'MinuteVent_r',
                    'DeliveredTidalVolume_r', 'FiO2Set_r', 'PEEPSet_r', 'PIP_r', 'PlateauPres_r', 'RAW_r',
                    'RSBI_r', 'RSBIRate_r', 'mSBP_r', 'mDBP_r', 'mMAP_r', 'CV_HR_r', 'mCrdIndx_r', 'mCVP_r', 'Art_BE_r',
                    'Art_CO2_r', 'Art_PaCO2_r', 'Art_PaO2_r', 'Art_pH_r', 'Na_r', 'K_r', 'Cl_r', 'Glucose_r', 'Ca_r', 'Mg_r',
                    'IonCa_r', 'Lactate_r', 'GCS_r', 'temp_r', 'Age_r',
                    'Antiarrhythmic_m', 'Anticoagulant_m', 'Antiplatelet_m', 'Benzodiazepine_m', 'beta_Blocking_m', 
                    'Calcium_channel_blocking_m', 'Diuretic_m', 'Hemostatic_m', 'Inotropic_m', 'Insulin_m', 
                    'Nondepolarizing_m', 'sedatives_m', 'Somatostatin_preparation_m', 'Sympathomimetic_m', 
                    'Thrombolytic_m', 'Vasodilating_m', 'AIDS_p', 'HemMalig_p', 'MetCarcinoma_p', 'medtype.label', 
                    'location.label']

                # for readmission, 'died' refers to 'target'
                # vns = ['died', 'sid', 'timeindex', 'overall_l', 'overall_m', 'overall_d', 'kidney_l', 'lvr_l',
                # 'hemat_l', 'lung_l', 'cardio_l', 'acidbase_l', 'elytes_l', 'general_l', 'kidney_m', 'lvr_m',
                # 'hemat_m', 'lung_m', 'cardio_m', 'acidbase_m', 'elytes_m', 'general_m', 'kidney_d', 'lvr_d',
                # 'hemat_d', 'lung_d', 'cardio_d', 'acidbase_d', 'elytes_d', 'general_d',
                # 'Creatinine', 'BUN', 'BUNtoCr', 'urineByHrByWeight', 'eGFR', 'AST', 'ALT', 'TBili', 'DBili',
                # 'Albumin', 'tProtein', 'ASTtoALT', 'HCT', 'Hgb', 'INR', 'Platelets', 'PT', 'PTT', 'RBC', 'WBC',
                # 'RESP', 'mSaO2', 'PaO2toFiO2', 'MinuteVent', 'DeliveredTidalVolume', 'FiO2Set', 'PEEPSet', 'PIP',
                # 'PlateauPres', 'RAW', 'RSBI', 'RSBIRate', 'mSBP', 'mDBP', 'mMAP', 'CV_HR', 'mCrdIndx', 'mCVP',
                # 'Art_BE', 'Art_CO2', 'Art_PaCO2', 'Art_PaO2', 'Art_pH', 'Na', 'K', 'Cl', 'Glucose', 'Ca', 'Mg', 
                # 'IonCa', 'Lactate', 'GCS', 'temp', 'Age',
                # 'Creatinine_n', 'BUN_n', 'BUNtoCr_n', 'urineByHrByWeight_n', 'eGFR_n', 'AST_n', 'ALT_n', 'TBili_n',
                # 'DBili_n', 'Albumin_n', 'tProtein_n', 'ASTtoALT_n', 'HCT_n', 'Hgb_n', 'INR_n', 'Platelets_n', 'PT_n',
                # 'PTT_n', 'RBC_n', 'WBC_n', 'RESP_n', 'mSaO2_n', 'PaO2toFiO2_n', 'MinuteVent_n', 'DeliveredTidalVolume_n',
                # 'FiO2Set_n', 'PEEPSet_n', 'PIP_n', 'PlateauPres_n', 'RAW_n', 'RSBI_n', 'RSBIRate_n', 'mSBP_n', 'mDBP_n',
                # 'mMAP_n', 'CV_HR_n', 'mCrdIndx_n', 'mCVP_n', 'Art_BE_n', 'Art_CO2_n', 'Art_PaCO2_n', 'Art_PaO2_n',
                # 'Art_pH_n', 'Na_n', 'K_n', 'Cl_n', 'Glucose_n', 'Ca_n', 'Mg_n', 'IonCa_n', 'Lactate_n', 'GCS_n',
                # 'temp_n', 'Age_n', 
                # 'Antiarrhythmic_m', 'Anticoagulant_m', 'Antiplatelet_m', 'Benzodiazepine_m', 'beta_Blocking_m', 
                # 'Calcium_channel_blocking_m', 'Diuretic_m', 'Hemostatic_m', 'Inotropic_m', 'Insulin_m', 
                # 'Nondepolarizing_m', 'sedatives_m', 'Somatostatin_preparation_m', 'Sympathomimetic_m', 
                # 'Thrombolytic_m', 'Vasodilating_m', 'AIDS_p', 'HemMalig_p', 'MetCarcinoma_p', 'medtype.label', 
                # 'location.label']

                # vns = ['died', 'sid', 'time', 'overall_l', 'overall_m', 'overall_d', 'kidney_l', 'lvr_l',
                # 'hemat_l', 'lung_l', 'cardio_l', 'acidbase_l', 'elytes_l', 'general_l', 'kidney_m',
                # 'lvr_m', 'hemat_m', 'lung_m', 'cardio_m', 'acidbase_m', 'elytes_m', 'general_m',
                # 'kidney_d', 'lvr_d', 'hemat_d', 'lung_d', 'cardio_d', 'acidbase_d', 'elytes_d', 'general_d'
                # 'Creatinine', 'BUN', 'BUNtoCr', 'urineByHrByWeight', 'eGFR', 'AST', 'ALT', 'TBili', 'DBili',
                # 'Albumin', 'tProtein', 'ASTtoALT', 'HCT', 'Hgb', 'INR',
                # 'Platelets', 'PT', 'PTT', 'RBC', 'WBC', 'RESP', 'mSaO2', 'PaO2toFiO2', 'MinuteVent',
                # 'DeliveredTidalVolume', 'FiO2Set', 'PEEPSet', 'PIP', 'PlateauPres', 'RAW',
                # 'RSBI', 'RSBIRate', 'mSBP', 'mDBP', 'mMAP', 'CV_HR', 'mCrdIndx', 'mCVP', 'Art_BE',
                # 'Art_CO2', 'Art_PaCO2', 'Art_PaO2', 'Art_pH', 'Na', 'K', 'Cl', 'Glucose', 'Ca', 'Mg',
                # 'IonCa', 'Lactate', 'GCS', 'temp', 'Age', 'Creatinine_n', 'BUN_n', 'BUNtoCr_n',
                # 'urineByHrByWeight_n', 'eGFR_n', 'AST_n', 'ALT_n', 'TBili_n', 'DBili_n',
                # 'Albumin_n', 'tProtein_n', 'ASTtoALT_n', 'HCT_n', 'Hgb_n', 'INR_n',
                # 'Platelets_n', 'PT_n', 'PTT_n', 'RBC_n', 'WBC_n', 'RESP_n', 'mSaO2_n',
                # 'PaO2toFiO2_n', 'MinuteVent_n', 'DeliveredTidalVolume_n', 'FiO2Set_n',
                # 'PEEPSet_n', 'PIP_n', 'PlateauPres_n', 'RAW_n', 'RSBI_n', 'RSBIRate_n',
                # 'mSBP_n', 'mDBP_n', 'mMAP_n', 'CV_HR_n', 'mCrdIndx_n', 'mCVP_n', 'Art_BE_n',
                # 'Art_CO2_n', 'Art_PaCO2_n', 'Art_PaO2_n', 'Art_pH_n', 'Na_n', 'K_n', 'Cl_n',
                # 'Glucose_n', 'Ca_n', 'Mg_n', 'IonCa_n', 'Lactate_n', 'GCS_n', 'temp_n', 'Age_n',
                # 'Antiarrhythmic', 'Anticoagulant', 'Antiplatelet', 'Benzodiazepine',
                # 'beta_Blocking', 'Calcium_channel_blocking', 'Diuretic', 'Hemostatic',
                # 'Inotropic', 'Insulin', 'Nondepolarizing', 'sedatives',
                # 'Somatostatin_preparation', 'Sympathomimetic', 'Thrombolytic',
                # 'Vasodilating', 'AIDS', 'HemMalig', 'MetCarcinoma', 'medtype', 'location', 'target']
                # print vns
                _ = row
            else:
                for i in range(len(row)):
                    if row[i] != "":
                        row[i] = float(row[i])
                    else:
                        row[i] = NaN
                    hv[vns[i]] = row[i]

                time = int(hv['timeindex']); sid = int(hv['sid'])
                if sid in partialPts:
                    continue
                if current_sid != "" and current_sid != sid:
                    if len(hmeans) == 0:
                        [tgraph, tnode] = self.interpolating(ptarr, current_sid,cz)
                    else:
                        [tgraph, tnode] = self.interpolating_raw(ptarr, current_sid, hmeans, hstds)
                    if len(tgraph) == 0 and current_mort == 1:
                        print("empty graph for %d" % (current_sid))
                    gcnt = self.write_graphs(fgraph, fnp, tgraph, tnode, current_sid)
                    current_sid = sid; ptarr = None

                # print hv

                if cz:
                    arow = array([time,
                        hv['Creatinine_n'], hv['BUN_n'], hv['BUNtoCr_n'], hv['urineByHrByWeight_n'], hv['eGFR_n'], hv['AST_n'], hv['ALT_n'], hv['TBili_n'], hv['DBili_n'], hv['Albumin_n'], hv['tProtein_n'], hv['ASTtoALT_n'], hv['HCT_n'], hv['Hgb_n'], hv['INR_n'], hv['Platelets_n'], hv['PT_n'], hv['PTT_n'], hv['RBC_n'], hv['WBC_n'], hv['RESP_n'], hv['mSaO2_n'], hv['PaO2toFiO2_n'], hv['MinuteVent_n'], hv['DeliveredTidalVolume_n'], hv['FiO2Set_n'], hv['PEEPSet_n'], hv['PIP_n'], hv['RSBI_n'], hv['RSBIRate_n'], hv['RAW_n'], hv['PlateauPres_n'], hv['mSBP_n'], hv['mDBP_n'], hv['mMAP_n'], hv['CV_HR_n'], hv['mCrdIndx_n'], hv['mCVP_n'], hv['Art_BE_n'], hv['Art_CO2_n'], hv['Art_PaCO2_n'], hv['Art_PaO2_n'], hv['Art_pH_n'], hv['Na_n'], hv['K_n'], hv['Cl_n'], hv['Glucose_n'], hv['Ca_n'], hv['Mg_n'], hv['IonCa_n'], hv['Lactate_n'], hv['GCS_n'], hv['temp_n'],
                        # meds
                        hv['Antiarrhythmic_m'], hv['Anticoagulant_m'], hv['Antiplatelet_m'], hv['Benzodiazepine_m'], hv['beta_Blocking_m'], hv['Calcium_channel_blocking_m'], hv['Diuretic_m'], hv['Hemostatic_m'], hv['Inotropic_m'], hv['Insulin_m'], hv['Nondepolarizing_m'], hv['sedatives_m'], hv['Somatostatin_preparation_m'], hv['Sympathomimetic_m'], hv['Thrombolytic_m'], hv['Vasodilating_m'],
                        # malig
                        hv['AIDS_p'], hv['HemMalig_p'], hv['MetCarcinoma_p'],
                        # 
                        hv['medtype.label'], hv['location.label']])
                else:
                    arow = array([time,
                        hv['Creatinine_r'], hv['BUN_r'], hv['BUNtoCr_r'], hv['urineByHrByWeight_r'], hv['eGFR_r'], hv['AST_r'], hv['ALT_r'], hv['TBili_r'], hv['DBili_r'], hv['Albumin_r'], hv['tProtein_r'], hv['ASTtoALT_r'], hv['HCT_r'], hv['Hgb_r'], hv['INR_r'], hv['Platelets_r'], hv['PT_r'], hv['PTT_r'], hv['RBC_r'], hv['WBC_r'], hv['RESP_r'], hv['mSaO2_r'], hv['PaO2toFiO2_r'], hv['MinuteVent_r'], hv['DeliveredTidalVolume_r'], hv['FiO2Set_r'], hv['PEEPSet_r'], hv['PIP_r'], hv['RSBI_r'], hv['RSBIRate_r'], hv['RAW_r'], hv['PlateauPres_r'], hv['mSBP_r'], hv['mDBP_r'], hv['mMAP_r'], hv['CV_HR_r'], hv['mCrdIndx_r'], hv['mCVP_r'], hv['Art_BE_r'], hv['Art_CO2_r'], hv['Art_PaCO2_r'], hv['Art_PaO2_r'], hv['Art_pH_r'], hv['Na_r'], hv['K_r'], hv['Cl_r'], hv['Glucose_r'], hv['Ca_r'], hv['Mg_r'], hv['IonCa_r'], hv['Lactate_r'], hv['GCS_r'], hv['temp_r'], 
                        # meds
                        hv['Antiarrhythmic_m'], hv['Anticoagulant_m'], hv['Antiplatelet_m'], hv['Benzodiazepine_m'], hv['beta_Blocking_m'], hv['Calcium_channel_blocking_m'], hv['Diuretic_m'], hv['Hemostatic_m'], hv['Inotropic_m'], hv['Insulin_m'], hv['Nondepolarizing_m'], hv['sedatives_m'], hv['Somatostatin_preparation_m'], hv['Sympathomimetic_m'], hv['Thrombolytic_m'], hv['Vasodilating_m'],
                        # malig
                        hv['AIDS_p'], hv['HemMalig_p'], hv['MetCarcinoma_p'],
                        #
                        # hv['Creatinine_n'], hv['BUN_n'], hv['BUNtoCr_n'], hv['urineByHrByWeight_n'], hv['eGFR_n'], hv['AST_n'], hv['ALT_n'], hv['TBili_n'], hv['DBili_n'], hv['Albumin_n'], hv['tProtein_n'], hv['ASTtoALT_n'], hv['HCT_n'], hv['Hgb_n'], hv['INR_n'], hv['Platelets_n'], hv['PT_n'], hv['PTT_n'], hv['RBC_n'], hv['WBC_n'], hv['RESP_n'], hv['mSaO2_n'], hv['PaO2toFiO2_n'], hv['MinuteVent_n'], hv['DeliveredTidalVolume_n'], hv['FiO2Set_n'], hv['PEEPSet_n'], hv['PIP_n'], hv['RSBI_n'], hv['RSBIRate_n'], hv['RAW_n'], hv['PlateauPres_n'], hv['mSBP_n'], hv['mDBP_n'], hv['mMAP_n'], hv['CV_HR_n'], hv['mCrdIndx_n'], hv['mCVP_n'], hv['Art_BE_n'], hv['Art_CO2_n'], hv['Art_PaCO2_n'], hv['Art_PaO2_n'], hv['Art_pH_n'], hv['Na_n'], hv['K_n'], hv['Cl_n'], hv['Glucose_n'], hv['Ca_n'], hv['Mg_n'], hv['IonCa_n'], hv['Lactate_n'], hv['GCS_n'], hv['temp_n'],
                        hv['medtype.label'], hv['location.label']])
                arow = arow.reshape(1, arow.size)
                if ptarr is None:
                    ptarr = arow
                else:
                    ptarr = vstack((ptarr,arow))

                current_sid = sid; current_mort = int(hv['readmit'])
        if len(hmeans) == 0:
            [tgraph, tnode] = self.interpolating(ptarr, current_sid, cz)
        else:
            [tgraph, tnode] = self.interpolating_raw(ptarr, current_sid, hmeans, hstds)
        gcnt = self.write_graphs(fgraph, fnp, tgraph, tnode, current_sid)


        fgraph.close()        
        fin.close()
        fnp.close()
        return;

