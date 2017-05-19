library("Rcpp")
library("mice")
library(parallel)
# library("Rcpp", lib.loc="/Users/XueY/Documents/R-packages")
# library("mice", lib.loc="/Users/XueY/Documents/R-packages")

# ******************* #
# USE SEED!!!!!!!!!!! #
# ******************* #

nRecordsPerCluster = 8
multiImp = 5
testSize = 234
minp = 0
minc = 0
nthfold = 0
ft = 'raw'
# mp_list = list("0.4","0.5","0.65")
# mc_list = list("0.4","0.6")
# mp_list = list("0.4","0.5","0.65","0")
# mc_list = list("0.1","0.3","0.4","0.6")
# mp_list = list("0")
# mc_list = list("0.4","0.6")
# mp_list = list("0.5","0.65")
# mc_list = list("0.6","0.7")
mp_list = list("0.5")
mc_list = list("0.6")

exclude_features = c('readmit', 'sid', 'timeindex', 
'Antiarrhythmic_agent', 'Anticoagulant', 'Antiplatelet_agent', 
'Benzodiazepine', 'beta.Blocking_agent', 'Calcium_channel_blocking_agent', 'Diuretic', 'Hemostatic_agent', 
'Inotropic_agent', 'Insulin', 'Nondepolarizing_agent', 'sedatives', 'Somatostatin_preparation', 
'Sympathomimetic_agent', 'Thrombolytic_agent', 'Vasodilating_agent', 'AIDS', 'HemMalig', 'MetCarcinoma', 
'medtype.label', 'location.label')

expri_list <- list()
for (mc in mc_list) {
	for (mp in mp_list) {
		# if (mp != "0.5" || mc != "0.6") {
		for (i in 0:4) {
			expri_list[[length(expri_list)+1]] <- list(mp,mc,i)
		}
		# }
	}
}

imputeTrain <- function(k) {
	params = expri_list[[k]]
	tmp = params[[1]]
	tmc = params[[2]]
	foldi = params[[3]]
	print('impute train')
	# print(tmp)
	# print(tmc)
	# print(foldi)

	# imputes raw
	# if (!dir.exists(sprintf('../data/seed2222/%s/mice/mp%s_mc%s',ft,tmp,tmc))) {
	# 	dir.create(sprintf('../data/seed2222/%s/mice/mp%s_mc%s',ft,tmp,tmc))
	# }
	# ftr = sprintf('../data/seed2222/%s/train_fold%d.csv',ft,foldi)
	# fout = sprintf('../data/seed2222/%s/mice/mp%s_mc%s/train_fold%d',ft,tmp,tmc,foldi)
	
	# imputes standardize_z
	# if (!dir.exists(sprintf('../data/seed2222/%s/standardize_z_mice/mp%s_mc%s',ft,tmp,tmc))) {
	# 	dir.create(sprintf('../data/seed2222/%s/standardize_z_mice/mp%s_mc%s',ft,tmp,tmc))
	# }
	# ftr = sprintf('../data/seed2222/%s/train_fold%d_z.csv',ft,foldi)
	# fout = sprintf('../data/seed2222/%s/standardize_z_mice/mp%s_mc%s/train_fold%d_z',ft,tmp,tmc,foldi)
	
	# imputes raw (one hour interval)
	# if (!dir.exists(sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s',ft,tmp,tmc))) {
	# 	dir.create(sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s',ft,tmp,tmc))
	# }
	# ftr = sprintf('../data/seed2222_one_hour_interval/%s/train_fold%d.csv',ft,foldi)
	# fout = sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s/train_fold%d',ft,tmp,tmc,foldi)
	
	# imputes standardize_z (one hour interval)
	# if (!dir.exists(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s',ft,tmp,tmc))) {
	# 	dir.create(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s',ft,tmp,tmc))
	# }
	# ftr = sprintf('../data/seed2222_one_hour_interval/%s/train_fold%d_z.csv',ft,foldi)
	# fout = sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/train_fold%d_z',ft,tmp,tmc,foldi)
	
	# imputes interped data
	if (!dir.exists(sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s',ft,tmp,tmc))) {
		dir.create(sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s',ft,tmp,tmc))
	}
	ftr = sprintf('../data/seed2222/%s/interp/train_fold%d.csv',ft,foldi)
	fout = sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s/dataset/train_fold%d',ft,tmp,tmc,foldi)
	
	print(fout)
	# ftr = sprintf('../data/train/alldata_readmit_last12h_X_train_fold%d.csv',foldi)
	# fout = sprintf('../data/train/last12h_pmm_X_mp%s_mc%s_fold%d',tmp,tmc,foldi)
	tr = read.csv(ftr, quote="\"")
	# cols = colnames(tr)
	# sid_l = tr$sid
	# tr$sid <- NULL
	imputed = mice(data=tr, m=multiImp, 
		pred=quickpred(tr,minpuc=as.numeric(tmp),mincor=as.numeric(tmc),
			exclude=exclude_features), printFlag=FALSE)
	# print(quickpred(tr,minpuc=as.numeric(tmp),mincor=as.numeric(tmc),exclude=exclude_features))
	combineImputed(imputed, fout)
}

imputeTest <- function(k) {
	params = expri_list[[k]]
	tmp = params[[1]]
	tmc = params[[2]]
	foldi = params[[3]]
	print('impute test')
	print(tmp)
	print(tmc)
	print(foldi)

	# imputed_tr = read.csv(sprintf('../data/seed2222/z/mice/train_mp%s_mc%s_fold%d.csv',tmp,tmc,foldi), quote="\"")
	imputed_tr = read.csv(sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s/dataset/train_fold%d.csv',ft,tmp,tmc,foldi), quote="\"")

	if (!dir.exists(sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s',ft,foldi,tmp,tmc))) {
		dir.create(sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s',ft,foldi,tmp,tmc))
	}

	conn <- file(sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/sid_list.txt',ft,foldi),open='r')
	linn <- readLines(conn)
	# linn <- linn[c(1,3,5)]
	for (sid in linn) {
		print(sid)
		te = read.csv(sprintf("../data/seed2222/%s/interp/mice/imputing/test_fold%d/%s.csv",ft,foldi,sid), quote="\"")
		if (sum(is.na(te)) == 0) {
			write.csv(te, sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s/%s.csv',ft,foldi,tmp,tmc,sid), row.names=FALSE)
		}
		else {
			toImp = rbind(imputed_tr,te)
			imputed = mice(data=toImp, m=multiImp, 
				pred=quickpred(toImp,minpuc=as.numeric(tmp),
					mincor=as.numeric(tmc),exclude=exclude_features), printFlag=FALSE)
			fout = sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s/%s',ft,foldi,tmp,tmc,sid)
			combineImputed(imputed, fout, sid=as.numeric(sid), test=TRUE)
		}
	}
	collect_test_records(foldi,tmp,tmc,linn)
	close(conn)
}

combineImputed <- function(imputed, fn, sid=0, test=FALSE) {
	print('combining imputed')
	if (!dir.exists(fn)) {
		dir.create(fn)
	}

	combined <- complete(imputed,1)
	if (test) {
		combined <- combined[combined$sid==sid,]
	}
	write.csv(combined, paste(fn, sprintf("/imp%d.csv",1),sep=""), row.names=FALSE)

	for (i in 2:multiImp) {
		cp <- complete(imputed,i)
		if (test) {
			cp <- cp[cp$sid==sid,]
		}
		write.csv(cp, paste(fn, sprintf("/imp%d.csv",i),sep=""), row.names=FALSE)
		
		combined <- combined + cp
	}
	combined <- combined / 5
	write.csv(combined, paste(fn,".csv",sep=""), row.names=FALSE)
}

imputeTest_per_patient <- function(x) {
	# print(x)
	# cat(sprintf('%s,%s,%d: %s\n',minp,minc,nthfold,x))
	
	# raw
	# te = read.csv(sprintf("../data/seed2222/%s/mice/imputing/test_fold%d/%s.csv",ft,nthfold,x), quote="\"")
	# fout = sprintf('../data/seed2222/%s/mice/imputing/test_fold%d/mp%s_mc%s/%s',ft,nthfold,minp,minc,x)
	
	# z
	# te = read.csv(sprintf("../data/seed2222/%s/standardize_z_mice/imputing/test_fold%d_z/%s.csv",ft,nthfold,x), quote="\"")
	# fout = sprintf('../data/seed2222/%s/standardize_z_mice/imputing/test_fold%d_z/mp%s_mc%s/%s',ft,nthfold,minp,minc,x)
	
	# raw (one hour interval)
	# te = read.csv(sprintf("../data/seed2222_one_hour_interval/%s/mice/imputing/test_fold%d/%s.csv",ft,nthfold,x), quote="\"")
	# fout = sprintf('../data/seed2222_one_hour_interval/%s/mice/imputing/test_fold%d/mp%s_mc%s/%s',ft,nthfold,minp,minc,x)
	
	# z (one hour interval)
	# te = read.csv(sprintf("../data/seed2222_one_hour_interval/%s/standardize_z_mice/imputing/test_fold%d/%s.csv",ft,nthfold,x), quote="\"")
	# fout = sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/imputing/test_fold%d/mp%s_mc%s/%s',ft,nthfold,minp,minc,x)

	# print(fout)
	# toImp = rbind(imputed_tr,te)
	# imputed = mice(data=toImp, m=multiImp, 
	# 	pred=quickpred(toImp,minpuc=as.numeric(minp),
	# 		mincor=as.numeric(minc),exclude=exclude_features), printFlag=FALSE)
	# combineImputed(imputed, fout, sid=as.numeric(x), test=TRUE)
	te = read.csv(sprintf("../data/seed2222/%s/interp/mice/imputing/test_fold%d/%s.csv",ft,foldi,sid), quote="\"")
	if (sum(is.na(te)) == 0) {
		write.csv(te, sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s/%s.csv',ft,foldi,tmp,tmc,sid), row.names=FALSE)
	}
	else {
		toImp = rbind(imputed_tr,te)
		imputed = mice(data=toImp, m=multiImp, 
			pred=quickpred(toImp,minpuc=as.numeric(tmp),
				mincor=as.numeric(tmc),exclude=exclude_features), printFlag=FALSE)
		fout = sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s/%s',ft,foldi,tmp,tmc,sid)
		combineImputed(imputed, fout, sid=as.numeric(sid), test=TRUE)
	}
}

collect_test_records <- function(i,minp,minc,sid_list) {
	# folder <- sprintf('../data/seed2222/%s/mice/imputing/test_fold%d/mp%s_mc%s',ft,i,minp,minc)
	# folder <- sprintf('../data/seed2222/%s/standardize_z_mice/imputing/test_fold%d_z/mp%s_mc%s',ft,i,minp,minc)

	# folder <- sprintf('../data/seed2222_one_hour_interval/%s/mice/imputing/test_fold%d/mp%s_mc%s',ft,i,minp,minc)
	folder <- sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s',ft,i,minp,minc)
	

	if (dir.exists(folder)) {
		combined <- data.frame()
		for (sid in sid_list) {
			f <- sprintf('%s/%s.csv',folder,sid)
			if (file.exists(f)) {
				df <- read.csv(f, quote="\"")
				combined <- rbind(combined, df)
			}
			else {
				print('no imputed file.')
			}
		}
		# raw
		# if (!dir.exists(sprintf('../data/seed2222/%s/mice/mp%s_mc%s',ft,minp,minc))) {
		# 	dir.create(sprintf('../data/seed2222/%s/mice/mp%s_mc%s',ft,minp,minc))
		# }
		# write.csv(combined, sprintf('../data/seed2222/z/mice/test_mp%s_mc%s_fold%d.csv',minp,minc,i), row.names=FALSE)
		
		# z
		# if (!dir.exists(sprintf('../data/seed2222/%s/standardize_z_mice/mp%s_mc%s',ft,minp,minc))) {
		# 	dir.create(sprintf('../data/seed2222/%s/standardize_z_mice/mp%s_mc%s',ft,minp,minc))
		# }
		# write.csv(combined, sprintf('../data/seed2222/%s/standardize_z_mice/mp%s_mc%s/dataset/test_fold%d_z.csv',ft,minp,minc,i), row.names=FALSE)

		# raw (one hour interval)
		# if (!dir.exists(sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s',ft,minp,minc))) {
		# 	dir.create(sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s',ft,minp,minc))
		# }
		# write.csv(combined, sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s/dataset/test_fold%d.csv',ft,minp,minc,i), row.names=FALSE)
	
		# z (one hour interval)
		# if (!dir.exists(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s',ft,minp,minc))) {
		# 	dir.create(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s',ft,minp,minc))
		# }
		# write.csv(combined, sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/dataset/test_fold%d_z.csv',ft,minp,minc,i), row.names=FALSE)
		
		# interp
		if (!dir.exists(sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s',ft,minp,minc))) {
			dir.create(sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s',ft,minp,minc))
		}
		write.csv(combined, sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s/dataset/test_fold%d.csv',ft,minp,minc,i), row.names=FALSE)
	
	}
	else {
		print('no imputed folder.')
	}
}

# imputeTest(2)
# for (i in 1:length(expri_list)) {
# 	imputeTrain(i)
# 	imputeTest(i)
# }

n_cores <- detectCores()

# imputeTrain(1)
mclapply(1:length(expri_list), imputeTrain, mc.cores = 5)
# mclapply(1:length(expri_list), imputeTest, mc.cores = n_cores)

# conn <- file(sprintf('../data/seed2222/%s/mice/imputing/test_fold%d/sid_list.txt',ft,0),open='r')
# linn <- readLines(conn)
# collect_test_records(0,"0.5","0.6",linn)
# close(conn)

for (mc in mc_list) {
	for (mp in mp_list) {
		# if (mp != "0.5" || mc != "0.6") {
		for (i in 0:4) {
			# print(mp)
			# print(mc)
			# print(i)
			minp = mp
			minc = mc
			nthfold = i

			# raw (one hour interval)
			# if (!dir.exists(sprintf('../data/seed2222_one_hour_interval/%s/mice/imputing/test_fold%d/mp%s_mc%s',ft,nthfold,minp,minc))) {
			# 	dir.create(sprintf('../data/seed2222_one_hour_interval/%s/mice/imputing/test_fold%d/mp%s_mc%s',ft,nthfold,minp,minc))
			# }
			# imputed_tr = read.csv(sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s/dataset/train_fold%d.csv',ft,minp,minc,nthfold), quote="\"")
			# print(sprintf('../data/seed2222_one_hour_interval/%s/mice/mp%s_mc%s/dataset/train_fold%d.csv',ft,minp,minc,nthfold))
			# conn <- file(sprintf('../data/seed2222_one_hour_interval/%s/mice/imputing/test_fold%d/sid_list.txt',ft,nthfold),open='r')
			
			# z (one hour interval)
			# if (!dir.exists(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/imputing/test_fold%d/mp%s_mc%s',ft,nthfold,minp,minc))) {
			# 	dir.create(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/imputing/test_fold%d/mp%s_mc%s',ft,nthfold,minp,minc))
			# }
			# imputed_tr = read.csv(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/dataset/train_fold%d_z.csv',ft,minp,minc,nthfold), quote="\"")
			# print(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/dataset/train_fold%d_z.csv',ft,minp,minc,nthfold))
			# conn <- file(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/imputing/test_fold%d/sid_list.txt',ft,nthfold),open='r')
			
			# interp
			if (!dir.exists(sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s',ft,nthfold,minp,minc))) {
				dir.create(sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/mp%s_mc%s',ft,nthfold,minp,minc))
			}
			
			imputed_tr = read.csv(sprintf('../data/seed2222/%s/interp/mice/mp%s_mc%s/dataset/train_fold%d.csv',ft,minp,minc,nthfold), quote="\"")
			# print(sprintf('../data/seed2222_one_hour_interval/%s/standardize_z_mice/mp%s_mc%s/dataset/train_fold%d_z.csv',ft,minp,minc,nthfold))
			conn <- file(sprintf('../data/seed2222/%s/interp/mice/imputing/test_fold%d/sid_list.txt',ft,nthfold),open='r')
			

			linn <- readLines(conn)
			mclapply(linn, imputeTest_per_patient, mc.cores=n_cores)
			collect_test_records(nthfold,minp,minc,linn)
			close(conn)
		}
	}
}

# minp = 0.4
# minc = 0.6
# nthfold = 0
# imputed_tr = read.csv(sprintf('../data/seed2222/z/mice/train_mp%s_mc%s_fold%d.csv',minp,minc,nthfold), quote="\"")
# conn <- file(sprintf('../data/seed2222/z/mice/imputing/test_fold%d/sid_list.txt',nthfold),open='r')
# linn <- readLines(conn)
# mclapply(linn, imputeTest_per_patient, mc.cores=n_cores)
# collect_test_records(nthfold,minp,minc,linn)
# nthfold = 1
# imputed_tr = read.csv(sprintf('../data/seed2222/z/mice/train_mp%s_mc%s_fold%d.csv',minp,minc,nthfold), quote="\"")
# conn <- file(sprintf('../data/seed2222/z/mice/imputing/test_fold%d/sid_list.txt',nthfold),open='r')
# linn <- readLines(conn)
# mclapply(linn, imputeTest_per_patient, mc.cores=n_cores)
# collect_test_records(nthfold,minp,minc,linn)
# close(conn)

# nthfold = 2
# imputed_tr = read.csv(sprintf('../data/seed2222/z/mice/train_mp%s_mc%s_fold%d.csv',minp,minc,nthfold), quote="\"")
# conn <- file(sprintf('../data/seed2222/z/mice/imputing/test_fold%d/sid_list.txt',nthfold),open='r')
# linn <- readLines(conn)
# mclapply(linn, imputeTest_per_patient, mc.cores=n_cores)
# collect_test_records(nthfold,minp,minc,linn)
# close(conn)

# nthfold = 3
# imputed_tr = read.csv(sprintf('../data/seed2222/z/mice/train_mp%s_mc%s_fold%d.csv',minp,minc,nthfold), quote="\"")
# conn <- file(sprintf('../data/seed2222/z/mice/imputing/test_fold%d/sid_list.txt',nthfold),open='r')
# linn <- readLines(conn)
# mclapply(linn, imputeTest_per_patient, mc.cores=n_cores)
# collect_test_records(nthfold,minp,minc,linn)
# close(conn)

# nthfold = 4
# imputed_tr = read.csv(sprintf('../data/seed2222/z/mice/train_mp%s_mc%s_fold%d.csv',minp,minc,nthfold), quote="\"")
# conn <- file(sprintf('../data/seed2222/z/mice/imputing/test_fold%d/sid_list.txt',nthfold),open='r')
# linn <- readLines(conn)
# mclapply(linn, imputeTest_per_patient, mc.cores=n_cores)
# collect_test_records(nthfold,minp,minc,linn)
# close(conn)

# imputeTest(1)

# for (mc in mc_list) {
# 	for (mp in mp_list) {
# 		for (i in 0:4) {
# 			print(mp)
# 			print(mc)
# 			print(i)
# 			minp = mp
# 			minc = mc
# 			nthfold = i
# 			folder = sprintf('../data/seed2222/z/mice/imputing/test_fold%d/mp%s_mc%s',nthfold,minp,minc)
# 			if (!dir.exists(folder)) {
# 				dir.create(folder)
# 			}
# 			conn <- file(sprintf('../data/seed2222/z/mice/imputing/test_fold%d/sid_list.txt',nthfold),open='r')
# 			linn <- readLines(conn)
			
# 			mclapply(linn, imputeTest, mc.cores=n_cores)
# 		}
# 	}
# }

# collect imputed test data
# for (i in 0:4) {
# 	for (mp in mp_list) {
# 		for (mc in mc_list) {
# 			collect_test_records(i,mp,mc)
# 		}
# 	}
# }
