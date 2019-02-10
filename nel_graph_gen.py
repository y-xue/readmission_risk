"""
Generate graph database from csv files of MIMIC data per Rohit's output.
Modified by yxue - 05-18-2017
First created by yluo - 04/26/2013

"""

#! /usr/bin/python;
import csv
import re
import os
import glob
import shutil
import sys
import mimic_stats as ms
from numpy import *

class NelGraphGen(object):
    missing_kidney = 0
    missing_lvr = 0
    missing_hemat = 0
    missing_cardio = 0
    missing_acidbase = 0
    missing_general = 0
    missing_lung = 0
    missing_elytes = 0
    def add_time(self, ncnt, time, prev_time, prev_tidx, olab, prev_olab, labstr):
        "For now ignore the actual time"
        etext = "";
        ncnt += 1;
        qd = self.discretize_abs_time(time)
        labstr = re.sub(r' +', "_", labstr.rstrip(" "))
        ntext = "v %s olab_%s_%s\n" % (ncnt, olab, labstr) # , qd
        
        if prev_tidx != -1:
            dtime = self.discretize_duration(time - prev_time)
            pqd = self.discretize_abs_time(prev_time)
        # etext += "e %s %s t%s_%s\n" % (prev_tidx, ncnt, pqd, qd)
            if (prev_olab == olab):
                etext += "e %s %s tsame\n" % (prev_tidx, ncnt) #, dtime
            elif prev_olab < olab:
                etext += "e %s %s tup\n" % (prev_tidx, ncnt) #, dtime
            elif prev_olab > olab:
                etext += "e %s %s tdown\n" % (prev_tidx, ncnt) # , dtime
            else:
                sys.exit("not supposed to be here %s - %s." % (prev_olab, olab))
        return (ntext, etext, ncnt);
    
    def collapse_olab(self, olab):
        "The grouping is according to Rohit's cluster mortality correlation"
        olab = int(olab)
        return olab;
        # if olab <= 6:
        #     return 1 
        # elif olab <= 8:
        #     return 2 
        # elif olab <= 9:
        #     return 3 
        # else:
        #     return 4 
        
    def discretize_abs_time(self, time):
        # return "qd"
        return "qd_%s" % (time / 360)
    
    def discretize_duration(self, dur):
        # return "dur_%s" % dur
        # if dur < 60:
        #     return "1hdur"
        # elif dur < 120:
        #     return "2hdur"
        # el
        if dur < 180:
            return "3hdur"
        elif dur < 1440:
            return "mdur"
        else:
            return "ldur"

    def discretize_duration2(self, time):
        """ The following is based on plotting the progressive distribution of the
        changing time, in an equi-width fashion.
        0-30min: 5min step
        30-1h: 10min step
        1h-1.5h: 15min step
        1.5h-2h: 30min step
        2h-6h: 1h step 
        6h-12h: 2h step
        12-24h: 4h step
        24-48h: 8h step
        2d-3d: 1d step
        3d+: together"""
        if time < 30:
            return "5min_%s" % (time / 5)
        elif time < 60:
            return "10min_%s" % (time / 10)
        elif time < 90:
            return "15min_%s" % (time / 15)
        elif time < 120:
            return "30min_%s" % (time / 30)
        elif time < 360:
            return "1h_%s" % (time / 60)
        elif time < 720:
            return "2h_%s" % (time / 120)
        elif time < 1440:
            return "4h_%s" % (time / 240)
        elif time < 2880:
            return "8h_%s" % (time / 480)
        elif time < 4320:
            return "1d_%s" % (time / 1440)
        else:
            return "3dup"


    def add_magnitude(self, kidney_m, lvr_m, hemat_m, lung_m, cardio_m, acidbase_m, 
                      elytes_m, ncnt, tidx):
        ntext = ""; etext = ""
        if kidney_m != "":
            nlab = "kidney_m_%s" % (kidney_m)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab) 
            etext += "e %s %s mag\n" % (tidx, ncnt) 

        if lvr_m != "":
            nlab = "lvr_m_%s" % (lvr_m)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s mag\n" % (tidx, ncnt)

        if hemat_m != "":
            nlab = "hemat_m_%s" % (hemat_m)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s mag\n" % (tidx, ncnt)

        if lung_m != "":
            nlab = "lung_m_%s" % (lung_m)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s mag\n" % (tidx, ncnt)

        if cardio_m != "":
            nlab = "cardio_m_%s" % (cardio_m)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s mag\n" % (tidx, ncnt)

        if acidbase_m != "":
            nlab = "acidbase_m_%s" % (acidbase_m)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s mag\n" % (tidx, ncnt)

        if elytes_m != "":
            nlab = "elytes_m_%s" % (elytes_m)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s mag\n" % (tidx, ncnt)
        return (ntext, etext, ncnt);



    def add_direction(self, kidney_d, lvr_d, hemat_d, lung_d, cardio_d, acidbase_d, 
                      elytes_d, ncnt, tidx):
        ntext = ""; etext = ""
        if kidney_d != "":
            nlab = "kidney_d_%s" % (kidney_d)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab) 
            etext += "e %s %s d\n" % (tidx, ncnt)

        if lvr_d != "":
            nlab = "lvr_d_%s" % (lvr_d)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s d\n" % (tidx, ncnt)

        if hemat_d != "":
            nlab = "hemat_d_%s" % (hemat_d)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s d\n" % (tidx, ncnt); 

        if lung_d != "":
            nlab = "lung_d_%s" % (lung_d)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s d\n" % (tidx, ncnt)

        if cardio_d != "":
            nlab = "cardio_d_%s" % (cardio_d)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s d\n" % (tidx, ncnt)

        if acidbase_d != "":
            nlab = "acidbase_d_%s" % (acidbase_d)
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s d\n" % (tidx, ncnt)

        if elytes_d != "":
            nlab = "elytes_d_%s" % (elytes_d) 
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab)
            etext += "e %s %s d\n" % (tidx, ncnt)
        return (ntext, etext, ncnt);

    def add_node_nondup(self, lab, hnid):
        "lab - numeric lab"
        if not hnid.has_key(lab):
            hnid[lab] = len(hnid)
        ntext = "v %s %s\n" % (hnid[lab], lab) 

    def add_med_label(self, med_l, ncnt, tidx):
        ntext = ""; etext = ""
        if med_l != "":
            nlab = "mlab_%s" % (int(med_l))
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab) 
            etext += "e %s %s med\n" % (tidx, ncnt)
        return (ntext, etext, ncnt, med_l);

    def d_organlab(self, lab):
        if not isnan(lab):
            lab = int(lab)
        return lab;
        
        # if lab <= 3:
        #     return "123"
        # elif lab <= 6:
        #     return "456"
        # elif lab <= 8:
        #     return "78"
        # elif isnan(lab):
        #     return lab
        # else:
        #     sys.exit("not supposed to be here. lab: %d" % (lab))

    def add_label(self, kidney_l, lvr_l, hemat_l, lung_l, cardio_l, acidbase_l, 
                  elytes_l, general_l, ncnt, tidx, olab_stats):
        ntext = ""; etext = ""; complete = True; olab_cnt = 0
        if kidney_l != "":
            nlab = "kidney_l_%s" % ( self.d_organlab(kidney_l) )
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt)
        else:
            complete = False

        if lvr_l != "":
            nlab = "lvr_l_%s" % ( self.d_organlab(lvr_l) )
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt)
        else:
            complete = False

        if hemat_l != "":
            nlab = "hemat_l_%s" % ( self.d_organlab(hemat_l) )
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt)
        else:
            complete = False

        if lung_l != "":
            nlab = "lung_l_%s" % ( self.d_organlab(lung_l) )
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt); 
        else:
            complete = False

        if cardio_l != "":
            nlab = "cardio_l_%s" % ( self.d_organlab(cardio_l) )
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt)
        else:
            complete = False

        if acidbase_l != "":
            nlab = "acidbase_l_%s" % ( self.d_organlab(acidbase_l) )
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt)
        else:
            complete = False

        if elytes_l != "":
            nlab = "elytes_l_%s" % ( self.d_organlab(elytes_l) ) 
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt)
        else:
            complete = False

        if general_l != "":
            nlab = "general_l_%s" % ( self.d_organlab(general_l) ) 
            ncnt += 1; ntext += "v %s %s\n" % (ncnt, nlab); olab_cnt += 1 
            etext += "e %s %s l\n" % (tidx, ncnt)
        else:
            complete = False

        if olab_stats.has_key(olab_cnt):
            olab_stats[olab_cnt] += 1
        else:
            olab_stats[olab_cnt] = 1
        if olab_cnt > 4:
            complete = True
        else:
            complete = False
        return (ntext, etext, ncnt, complete);

    def organlab_str(self, kidney_l, lvr_l, hemat_l, lung_l, cardio_l, acidbase_l, 
                     elytes_l, general_l, time):
        ans = ""; qd = self.discretize_abs_time(time); olab_cnt = 0
        if kidney_l != "":
            nlab = "kidney_l_%s" % ( self.d_organlab(kidney_l) ); 
            ans += nlab + " "
            olab_cnt += 1
        else:
            self.missing_kidney += 1



        # if hemat_l != "":
        #     nlab = "hemat_l_%s" % ( self.d_organlab(hemat_l) ); 
        #     ans += nlab + " "
        #     olab_cnt += 1
        # else:
        #     self.missing_hemat += 1

        if lung_l != "":
            nlab = "lung_l_%s" % ( self.d_organlab(lung_l) ); 
            ans += nlab + " "
            olab_cnt += 1 
        else:
            self.missing_lung += 1

        if cardio_l != "":
            nlab = "cardio_l_%s" % ( self.d_organlab(cardio_l) ); 
            ans += nlab + " "
            olab_cnt += 1
        else:
            self.missing_cardio += 1



        # if elytes_l != "":
        #     nlab = "elytes_l_%s" % ( self.d_organlab(elytes_l) ); 
        #     ans += nlab + " "
        #     olab_cnt += 1
        # else:
        #     self.missing_elytes += 1

        # if general_l != "":
        #     nlab = "general_l_%s" % ( self.d_organlab(general_l) ); 
        #     ans += nlab + " "
        #     olab_cnt += 1
        # else:
        #     self.missing_general += 1

        if lvr_l != "":
            nlab = "lvr_l_%s" % ( self.d_organlab(lvr_l) ); 
            ans += nlab + " "
            # olab_cnt += 1
        else:
            self.missing_lvr += 1

        # if acidbase_l != "":
        #     nlab = "acidbase_l_%s" % ( self.d_organlab(acidbase_l) ); 
        #     ans += nlab + " "
        #     # olab_cnt += 1
        # else:
        #     self.missing_acidbase += 1 

        complete = True
        if olab_cnt < 3:
            complete = False
        return (ans, complete);

    def mag_dir_str(self, kidney_m, lvr_m, hemat_m, lung_m, cardio_m, acidbase_m, 
                    elytes_m, general_m,
                    kidney_d, lvr_d, hemat_d, lung_d, cardio_d, acidbase_d, 
                    elytes_d, general_d,
                    time):
        "Instead of adding organ labels as node, attach them after the overall label as bag of words"
        ans = ""; qd = discretize_abs_time(time)
        if kidney_m != "":
            nlab = "kidney_m_%s_%s" % (kidney_m, qd); ans += nlab + " "

        if lvr_m != "":
            nlab = "lvr_m_%s_%s" % (lvr_m, qd); ans += nlab + " "

        if hemat_m != "":
            nlab = "hemat_m_%s_%s" % (hemat_m, qd); ans += nlab + " "

        if lung_m != "":
            nlab = "lung_m_%s_%s" % (lung_m, qd); ans += nlab + " "

        if cardio_m != "":
            nlab = "cardio_m_%s_%s" % (cardio_m, qd); ans += nlab + " "

        if acidbase_m != "":
            nlab = "acidbase_m_%s_%s" % (acidbase_m, qd); ans += nlab + " "

        if elytes_m != "":
            nlab = "elytes_m_%s_%s" % (elytes_m, qd); ans += nlab + " "

        if general_m != "":
            nlab = "general_m_%s_%s" % (general_m, qd); ans += nlab + " "

        if kidney_d != "":
            nlab = "kidney_d_%s_%s" % (kidney_d, qd); ans += nlab + " "

        if lvr_d != "":
            nlab = "lvr_d_%s_%s" % (lvr_d, qd); ans += nlab + " "

        if hemat_d != "":
            nlab = "hemat_d_%s_%s" % (hemat_d, qd); ans += nlab + " "

        if lung_d != "":
            nlab = "lung_d_%s_%s" % (lung_d, qd); ans += nlab + " "

        if cardio_d != "":
            nlab = "cardio_d_%s_%s" % (cardio_d, qd); ans += nlab + " "

        if acidbase_d != "":
            nlab = "acidbase_d_%s_%s" % (acidbase_d, qd); ans += nlab + " "

        if elytes_d != "":
            nlab = "elytes_d_%s_%s" % (elytes_d, qd); ans += nlab + " "

        if general_d != "":
            nlab = "general_d_%s_%s" % (general_d, qd); ans += nlab + " "


        return ans;

    def attach_property(self, hp, hlab, time):
        ans = ""
        for k in hp.keys():
            if k == "loc":
                ans += "%s_%s:1 " % (k, hp[k])
            else:
                ans += "%s_%s:%f " % (k, hp[k], hlab[k]) # , int(time)
        return ans;
    

    def attach_med(self, Antiarrhythmic, Anticoagulant, Antiplatelet, Benzodiazepine, 
                   beta_Blocking, Calcium_channel_blocking, Diuretic, Hemostatic, 
                   Inotropic, Insulin, Nondepolarizing, sedatives, 
                   Somatostatin_preparation, Sympathomimetic, Thrombolytic, 
                   Vasodilating, time):
        ans = ""; qd = "qd" # discretize_abs_time(time)
        if int(Antiarrhythmic) != 0:
            ans += "med_Antiarrhythmic_%s " % (qd)

        if int(Anticoagulant) != 0:
            ans += "med_Anticoagulant_%s " % (qd)

        if int(Antiplatelet) != 0:
            ans += "med_Antiplatelet_%s " % (qd)

        if int(Benzodiazepine) != 0:
            ans += "med_Benzodiazepine_%s " % (qd) 

        if int(beta_Blocking) != 0:
            ans += "med_beta_Blocking_%s " % (qd)

        if int(Calcium_channel_blocking) != 0:
            ans += "med_Calcium_channel_blocking_%s " % (qd)

        if int(Diuretic) != 0:
            ans += "med_Diuretic_%s " % (qd)

        if int(Hemostatic) != 0:
            ans += "med_Hemostatic_%s " % (qd)

        if int(Inotropic) != 0:
            ans += "med_Inotropic_%s " % (qd)

        if int(Insulin) != 0:
            ans += "med_Insulin_%s " % (qd)

        if int(Nondepolarizing) != 0:
            ans += "med_Nondepolarizing_%s " % (qd)

        if int(sedatives) != 0:
            ans += "med_sedatives_%s " % (qd)

        if int(Somatostatin_preparation) != 0:
            ans += "med_Somatostatin_preparation_%s " % (qd)

        if int(Sympathomimetic) != 0:
            ans += "med_Sympathomimetic_%s " % (qd)

        if int(Thrombolytic) != 0:
            ans += "med_Thrombolytic_%s " % (qd)

        if int(Vasodilating) != 0:
            ans += "med_Vasodilating_%s " % (qd) 
        return ans;

    def d_measure(self, measure, raw=False):
        """Input is string, normal, within a sigma, beyond a sigma"""
        measure = float(measure)
        if raw:
            if measure <= 1 and measure >= -1:
                measure = 0
            elif measure <= 2 and measure > 1:
                measure = 1
            elif measure < -1 and measure >= -2:
                measure = -1
            elif measure > 2:
                measure = 2
            elif measure < -2:
                measure = -2
        else:
            if measure == 0:
                measure = 0
                # elif measure > 0:
                #     measure = int(math.ceil(measure))
                # elif measure < 0:
                #     measure = int(math.floor(measure))
            elif measure <= 1 and measure > 0:
                measure = 1
            elif measure < 0 and measure >= -1:
                measure = -1
            elif measure > 1:
                measure = 2
            elif measure < -1:
                measure = -2
        return measure;


    def d_measure_raw(self, measure):
        """Input is string, normal, within a sigma, beyond a sigma"""
        measure = float(measure)

        return measure;

    def attach_measure(self, Creatinine, BUN, BUNtoCr, urineByHrByWeight, eGFR, 
                       AST, ALT, TBili, DBili, Albumin, tProtein, 
                       ASTtoALT, HCT, Hgb, INR, Platelets, PT, PTT, 
                       RBC, WBC, RESP, mSaO2, PaO2toFiO2, MinuteVent, 
                       DeliveredTidalVolume, FiO2Set, PEEPSet, PIP, 
                       PlateauPres, RAW, RSBI, RSBIRate, mSBP, mDBP, 
                       mMAP, CV_HR, mCrdIndx, mCVP, Art_BE, Art_CO2, 
                       Art_PaCO2, Art_PaO2, Art_pH, Na, K, Cl, 
                       Glucose, Ca, Mg, IonCa, Lactate, GCS, temp, 
                       time):
        ans = ""; qd = "qd" # discretize_abs_time(time)
        if Creatinine != "":
            nlab = "Creatinine_%s_%s" % (self.d_measure(Creatinine), qd)
            ans += nlab + " "

        if BUN != "":
            nlab = "BUN_%s_%s" % (self.d_measure(BUN), qd); ans += nlab + " "

        if BUNtoCr != "":
            nlab = "BUNtoCr_%s_%s" % (self.d_measure(BUNtoCr), qd); ans += nlab + " "    

        if urineByHrByWeight != "":
            nlab = "urineByHrByWeight_%s_%s" % (self.d_measure(urineByHrByWeight), qd)
            ans += nlab + " "    

        if eGFR != "":
            nlab = "eGFR_%s_%s" % (self.d_measure(eGFR), qd); ans += nlab + " "    

        if AST != "":
            nlab = "AST_%s_%s" % (self.d_measure(AST), qd); ans += nlab + " "    

        if ALT != "":
            nlab = "ALT_%s_%s" % (self.d_measure(ALT), qd); ans += nlab + " "    

        if TBili != "":
            nlab = "TBili_%s_%s" % (self.d_measure(TBili), qd); ans += nlab + " "    

        if DBili != "":
            nlab = "DBili_%s_%s" % (self.d_measure(DBili), qd); ans += nlab + " "    

        if Albumin != "":
            nlab = "Albumin_%s_%s" % (self.d_measure(Albumin), qd); ans += nlab + " "    

        if tProtein != "":
            nlab = "tProtein_%s_%s" % (self.d_measure(tProtein), qd)
            ans += nlab + " "    

        if ASTtoALT != "":
            nlab = "ASTtoALT_%s_%s" % (self.d_measure(ASTtoALT), qd)
            ans += nlab + " "    

        if HCT != "":
            nlab = "HCT_%s_%s" % (self.d_measure(HCT), qd); ans += nlab + " "    

        if Hgb != "":
            nlab = "Hgb_%s_%s" % (self.d_measure(Hgb), qd); ans += nlab + " "    

        if INR != "":
            nlab = "INR_%s_%s" % (self.d_measure(INR), qd); ans += nlab + " "    

        if Platelets != "":
            nlab = "Platelets_%s_%s" % (self.d_measure(Platelets), qd)
            ans += nlab + " "    

        if PT != "":
            nlab = "PT_%s_%s" % (self.d_measure(PT), qd); ans += nlab + " "    

        if PTT != "":
            nlab = "PTT_%s_%s" % (self.d_measure(PTT), qd); ans += nlab + " "    

        if RBC != "":
            nlab = "RBC_%s_%s" % (self.d_measure(RBC), qd); ans += nlab + " "    

        if WBC != "":
            nlab = "WBC_%s_%s" % (self.d_measure(WBC), qd); ans += nlab + " "    

        if RESP != "":
            nlab = "RESP_%s_%s" % (self.d_measure(RESP), qd); ans += nlab + " "    

        if mSaO2 != "":
            nlab = "mSaO2_%s_%s" % (self.d_measure(mSaO2), qd); ans += nlab + " "    

        if PaO2toFiO2 != "":
            nlab = "PaO2toFiO2_%s_%s" % (self.d_measure(PaO2toFiO2), qd)
            ans += nlab + " "    

        if MinuteVent != "":
            nlab = "MinuteVent_%s_%s" % (self.d_measure(MinuteVent), qd)
            ans += nlab + " "    

        if DeliveredTidalVolume != "":
            nlab = "DeliveredTidalVolume_%s_%s" % (self.d_measure(DeliveredTidalVolume), qd); 
            ans += nlab + " "    

        if FiO2Set != "":
            nlab = "FiO2Set_%s_%s" % (self.d_measure(FiO2Set), qd); ans += nlab + " "    

        if PEEPSet != "":
            nlab = "PEEPSet_%s_%s" % (self.d_measure(PEEPSet), qd); ans += nlab + " "    

        if PIP != "":
            nlab = "PIP_%s_%s" % (self.d_measure(PIP), qd); ans += nlab + " "    

        if PlateauPres != "":
            nlab = "PlateauPres_%s_%s" % (self.d_measure(PlateauPres), qd)
            ans += nlab + " "    

        if RAW != "":
            nlab = "RAW_%s_%s" % (self.d_measure(RAW), qd); ans += nlab + " "    

        if RSBI != "":
            nlab = "RSBI_%s_%s" % (self.d_measure(RSBI), qd); ans += nlab + " "    

        if RSBIRate != "":
            nlab = "RSBIRate_%s_%s" % (self.d_measure(RSBIRate), qd)
            ans += nlab + " "    

        if mSBP != "":
            nlab = "mSBP_%s_%s" % (self.d_measure(mSBP), qd); ans += nlab + " "    

        if mDBP != "":
            nlab = "mDBP_%s_%s" % (self.d_measure(mDBP), qd); ans += nlab + " "    

        if mMAP != "":
            nlab = "mMAP_%s_%s" % (self.d_measure(mMAP), qd); ans += nlab + " "    

        if CV_HR != "":
            nlab = "CV_HR_%s_%s" % (self.d_measure(CV_HR), qd); ans += nlab + " "    

        if mCrdIndx != "":
            nlab = "mCrdIndx_%s_%s" % (self.d_measure(mCrdIndx), qd)
            ans += nlab + " "    

        if mCVP != "":
            nlab = "mCVP_%s_%s" % (self.d_measure(mCVP), qd); ans += nlab + " "    

        if Art_BE != "":
            nlab = "Art_BE_%s_%s" % (self.d_measure(Art_BE), qd); ans += nlab + " "    

        if Art_CO2 != "":
            nlab = "Art_CO2_%s_%s" % (self.d_measure(Art_CO2), qd); ans += nlab + " "    

        if Art_PaCO2 != "":
            nlab = "Art_PaCO2_%s_%s" % (self.d_measure(Art_PaCO2), qd)
            ans += nlab + " "    

        if Art_PaO2 != "":
            nlab = "Art_PaO2_%s_%s" % (self.d_measure(Art_PaO2), qd)
            ans += nlab + " "    

        if Art_pH != "":
            nlab = "Art_pH_%s_%s" % (self.d_measure(Art_pH), qd); ans += nlab + " "    

        if Na != "":
            nlab = "Na_%s_%s" % (self.d_measure(Na), qd); ans += nlab + " "    

        if K != "":
            nlab = "K_%s_%s" % (self.d_measure(K), qd); ans += nlab + " "    

        if Cl != "":
            nlab = "Cl_%s_%s" % (self.d_measure(Cl), qd); ans += nlab + " "    

        if Glucose != "":
            nlab = "Glucose_%s_%s" % (self.d_measure(Glucose), qd); ans += nlab + " "    

        if Ca != "":
            nlab = "Ca_%s_%s" % (self.d_measure(Ca), qd); ans += nlab + " "    

        if Mg != "":
            nlab = "Mg_%s_%s" % (self.d_measure(Mg), qd); ans += nlab + " "    

        if IonCa != "":
            nlab = "IonCa_%s_%s" % (self.d_measure(IonCa), qd); ans += nlab + " "    

        if Lactate != "":
            nlab = "Lactate_%s_%s" % (self.d_measure(Lactate), qd); ans += nlab + " "    

        if GCS != "":
            nlab = "GCS_%s_%s" % (self.d_measure(GCS), qd); ans += nlab + " "    

        if temp != "":
            nlab = "temp_%s_%s" % (self.d_measure(temp), qd); ans += nlab + " "    

        return ans;


    def add_medication(self, Antiarrhythmic, Anticoagulant, Antiplatelet, Benzodiazepine, 
                       beta_Blocking, Calcium_channel_blocking, Diuretic, 
                       Hemostatic, Inotropic, Insulin, Nondepolarizing, sedatives, 
                       Somatostatin_preparation, Sympathomimetic, Thrombolytic, 
                       Vasodilating, ncnt, tidx):
        ntext = ""; etext = ""; med_str = ""
        if Antiarrhythmic != 0:
            ncnt += 1; ntext += "v %s Antiarrhythmic\n" % (ncnt) 
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "Antiarrhythmic "

        if Anticoagulant != 0:
            ncnt += 1; ntext += "v %s Anticoagulant\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "Anticoagulant "

        if Antiplatelet != 0:
            ncnt += 1; ntext += "v %s Antiplatelet\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "Antiplatelet "

        # if Benzodiazepine != 0:
        #     ncnt += 1; ntext += "v %s Benzodiazepine\n" % (ncnt)
        #     etext += "e %s %s med\n" % (tidx, ncnt)
        #     med_str += "Benzodiazepine " 

        if beta_Blocking != 0:
            ncnt += 1; ntext += "v %s beta_Blocking\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "beta_Blocking "

        if Calcium_channel_blocking != 0:
            ncnt += 1; ntext += "v %s Calcium_channel_blocking\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "Calcium_channel_blocking "

        if Diuretic != 0:
            ncnt += 1; ntext += "v %s Diuretic\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "Diuretic "

        if Hemostatic != 0:
            ncnt += 1; ntext += "v %s Hemostatic\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "Hemostatic "

        if Inotropic != 0:
            ncnt += 1;  ntext += "v %s Inotropic\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)
            med_str += "Inotropic "

        if Insulin != 0:
            ncnt += 1; ntext += "v %s Insulin\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)     
            med_str += "Insulin "

        if Nondepolarizing != 0:
            ncnt += 1; ntext += "v %s Nondepolarizing\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)    
            med_str += "Nondepolarizing "

        # if sedatives != 0:
        #     ncnt += 1; ntext += "v %s sedatives\n" % (ncnt)
        #     etext += "e %s %s med\n" % (tidx, ncnt)  
        #     med_str += "sedatives "

        if Somatostatin_preparation != 0:
            ncnt += 1; ntext += "v %s Somatostatin_preparation\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)      
            med_str += "Somatostatin_preparation "

        # if Sympathomimetic != 0:
        #     ncnt += 1; ntext += "v %s Sympathomimetic\n" % (ncnt)
        #     etext += "e %s %s med\n" % (tidx, ncnt)  
        #     med_str += "Sympathomimetic "

        if Thrombolytic != 0:
            ncnt += 1; ntext += "v %s Thrombolytic\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)  
            med_str += "Thrombolytic "

        if Vasodilating != 0:
            ncnt += 1; ntext += "v %s Vasodilating\n" % (ncnt)
            etext += "e %s %s med\n" % (tidx, ncnt)  
            med_str += "Vasodilating " 
        return (ntext, etext, ncnt, med_str);
    
    def clr_gtxt(self, ntext, etext, nptext):
        for k in ntext.keys():
            ntext[k] = ""
            etext[k] = ""
            nptext[k] = ""

    def write_graphs(self, fgraph, fnp, ntext, etext, nptext, gcnt, sid):
        for k in ntext.keys():
            if ntext[k] != "":
                gcnt += 1
                fgraph.write("%s%sg %s\n\n" % (ntext[k], etext[k], gcnt))
                fnp.write("%sg %s %s\n\n" % (nptext[k], gcnt, sid))
        return gcnt;

    def scan_csv(self, fnin, fngraph, fnnp, fnmort, rec_t = 5):
        "TODO: discretize timeindex"
        fin = open(fnin, 'r')

        freader = csv.reader(fin, delimiter=',', quotechar="\"")
        fgraph = open(fngraph, 'w')
        fnp = open(fnnp, "w")

        # (hmort, hdays) = ms.read_mortality(fnmort)

        olab_stats = {}
        time_cut = 1440; time_lb = 1440
        lcnt = 0
        current_sid = ""
        ntext = ""; etext = ""; nptext = ""
        prev_tidx = -1; prev_olab = "-1"; prev_ncnt = 0; prev_med_str = ""
        ncnt = 0; gcnt = 0
        tidx = 0; prev_time = 0; max_time = 0; last_written = 0; only_low = True
        for row in freader:
            lcnt += 1
            if lcnt > 1:
                (died, sid, time, olab, overall_m, overall_d, kidney_l, lvr_l, 
                 hemat_l, lung_l, cardio_l, acidbase_l, elytes_l, general_l, 
                 kidney_m, lvr_m, hemat_m, lung_m, cardio_m, acidbase_m, elytes_m, 
                 general_m, kidney_d, lvr_d, hemat_d, lung_d, cardio_d, acidbase_d, 
                 elytes_d, general_d, Creatinine, BUN, BUNtoCr, urineByHrByWeight, 
                 eGFR, AST, ALT, TBili, DBili, Albumin, tProtein, ASTtoALT, HCT, 
                 Hgb, INR, 
                 Platelets, PT, PTT, RBC, WBC, RESP, mSaO2, PaO2toFiO2, MinuteVent,
                 DeliveredTidalVolume, FiO2Set, PEEPSet, PIP, PlateauPres, RAW, 
                 RSBI, RSBIRate, mSBP, mDBP, mMAP, CV_HR, mCrdIndx, mCVP, Art_BE, 
                 Art_CO2, Art_PaCO2, Art_PaO2, Art_pH, Na, K, Cl, Glucose, Ca, Mg, 
                 IonCa, Lactate, GCS, temp, Age, Creatinine_n, BUN_n, BUNtoCr_n, 
                 urineByHrByWeight_n, eGFR_n, AST_n, ALT_n, TBili_n, DBili_n, 
                 Albumin_n, tProtein_n, ASTtoALT_n, HCT_n, Hgb_n, INR_n, 
                 Platelets_n, PT_n, PTT_n, RBC_n, WBC_n, RESP_n, mSaO2_n, 
                 PaO2toFiO2_n, MinuteVent_n, DeliveredTidalVolume_n, FiO2Set_n, 
                 PEEPSet_n, PIP_n, PlateauPres_n, RAW_n, RSBI_n, RSBIRate_n, 
                 mSBP_n, mDBP_n, mMAP_n, CV_HR_n, mCrdIndx_n, mCVP_n, Art_BE_n, 
                 Art_CO2_n, Art_PaCO2_n, Art_PaO2_n, Art_pH_n, Na_n, K_n, Cl_n, 
                 Glucose_n, Ca_n, Mg_n, IonCa_n, Lactate_n, GCS_n, temp_n, Age_n, 
                 Antiarrhythmic, Anticoagulant, Antiplatelet, Benzodiazepine, 
                 beta_Blocking, Calcium_channel_blocking, Diuretic, Hemostatic, 
                 Inotropic, Insulin, Nondepolarizing, sedatives, 
                 Somatostatin_preparation, Sympathomimetic, Thrombolytic, 
                 Vasodilating, AIDS, HemMalig, MetCarcinoma, med_l, loc) = row
                # if not hmort.has_key(int(sid)):
                #     continue
                time = int(time); 
                olab = self.collapse_olab(olab)
                if current_sid != "" and current_sid != sid:
                    if ntext != "" and max_time >= time_lb: # and not only_low
                        fgraph.write("%s%sg %s\n\n" % (ntext, etext, current_sid))
                        fnp.write("%sg %s\n\n" % (nptext, current_sid))
                    current_sid = sid; max_time = 0; last_written = 0
                    ntext = ""; etext = ""; nptext = "";
                    ncnt = 0; prev_tidx = -1; 
                    prev_ncnt = 0; prev_olab = "-1"; prev_med_str = ""
                    only_low = True
                max_time = time
                current_sid = sid
                tidx = ncnt+1
                (labstr, complete) = self.organlab_str(kidney_l, lvr_l, hemat_l, lung_l, cardio_l, acidbase_l, elytes_l, general_l, time)
                (t_ntext, t_etext, ncnt) = self.add_time(ncnt, time, prev_time, prev_tidx, olab, prev_olab, labstr)
                # ptext = ("[%s]\n" % attach_label(kidney_l, lvr_l, hemat_l, lung_l, cardio_l, acidbase_l, elytes_l, general_l, kidney_m, lvr_m, hemat_m, lung_m, cardio_m, acidbase_m, elytes_m, general_m, kidney_d, lvr_d, hemat_d, lung_d, cardio_d, acidbase_d, elytes_d, general_d, time))

                ptext = ("[%s" % self.attach_med(Antiarrhythmic, Anticoagulant, Antiplatelet, Benzodiazepine, beta_Blocking, Calcium_channel_blocking, Diuretic, Hemostatic, Inotropic, Insulin, Nondepolarizing, sedatives, Somatostatin_preparation, Sympathomimetic, Thrombolytic, Vasodilating, time))

                ptext += ("%s]\n" % self.attach_measure(Creatinine_n, BUN_n, BUNtoCr_n, urineByHrByWeight_n, eGFR_n, AST_n, ALT_n, TBili_n, DBili_n, Albumin_n, tProtein_n, ASTtoALT_n, HCT_n, Hgb_n, INR_n, Platelets_n, PT_n, PTT_n, RBC_n, WBC_n, RESP_n, mSaO2_n, PaO2toFiO2_n, MinuteVent_n, DeliveredTidalVolume_n, FiO2Set_n, PEEPSet_n, PIP_n, PlateauPres_n, RAW_n, RSBI_n, RSBIRate_n, mSBP_n, mDBP_n, mMAP_n, CV_HR_n, mCrdIndx_n, mCVP_n, Art_BE_n, Art_CO2_n, Art_PaCO2_n, Art_PaO2_n, Art_pH_n, Na_n, K_n, Cl_n, Glucose_n, Ca_n, Mg_n, IonCa_n, Lactate_n, GCS_n, temp_n, time))

                # (inc_ntext, inc_etext, ncnt) = add_magnitude(kidney_m, lvr_m, hemat_m, lung_m, cardio_m, acidbase_m, elytes_m, ncnt, tidx)
                # ntext += inc_ntext; etext += inc_etext    

                # (inc_ntext, inc_etext, ncnt) = add_direction(kidney_d, lvr_d, hemat_d, lung_d, cardio_d, acidbase_d, elytes_d, ncnt, tidx)
                # ntext += inc_ntext; etext += inc_etext

                # (organlab_ntext, organlab_etext, ncnt, complete) = self.add_label(kidney_l, lvr_l, hemat_l, lung_l, cardio_l, acidbase_l, elytes_l, general_l, ncnt, tidx, olab_stats)


                # (med_ntext, med_etext, ncnt, med_str) = self.add_med_label(med_l, ncnt, tidx);
                # ans = add_medication(Antiarrhythmic, Anticoagulant, Antiplatelet, Benzodiazepine, beta_Blocking, Calcium_channel_blocking, Diuretic, Hemostatic, Inotropic, Insulin, Nondepolarizing, sedatives, Somatostatin_preparation, Sympathomimetic, Thrombolytic, Vasodilating, ncnt, tidx);
                # (med_ntext, med_etext, ncnt, med_str) = ans

                # only captures time points with changes (label or med)
                # and prev_med_str == med_str
                if (prev_olab == olab and time < time_cut): #  or not complete
                    ncnt = prev_ncnt; continue

                if last_written:
                    continue

                if olab != "1_2_3":
                    only_low = False
                ntext += t_ntext; etext += t_etext
                # ntext += organlab_ntext; etext += organlab_etext
                nptext += t_ntext; nptext += ptext
                # ntext += med_ntext; etext += med_etext

                prev_tidx = tidx; prev_olab = olab; prev_ncnt = ncnt
                # prev_med_str = med_str; 
                prev_time = time
                if time >= time_cut:
                    last_written = 1
        if ntext != "" and max_time >= time_lb: #  and not only_low
            fgraph.write("%s%sg %s\n\n" % (ntext, etext, current_sid))
            fnp.write("%sg %s\n\n" % (nptext, current_sid))

        # print("organ label count dist:")
        # for nlab in olab_stats.keys():
        #     print("%s %s" % (nlab, olab_stats[nlab]))
        print("missing kidney %d" % (self.missing_kidney))
        print("missing elytes %d" % (self.missing_elytes))
        print("missing general %d" % (self.missing_general))
        print("missing hemat %d" % (self.missing_hemat))
        print("missing lvr %d" % (self.missing_lvr))
        print("missing cardio %d" % (self.missing_cardio))
        print("missing lung %d" % (self.missing_lung))
        print("missing acidbase %d" % (self.missing_acidbase))
        fgraph.close()        
        fin.close()
        fnp.close()
        return;

