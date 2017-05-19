library("Rcpp")
library("mice")
library("miceadds")

myArgs <- commandArgs(trailingOnly = TRUE)
file=myArgs[1]
#me=myArgs[2]
folder=myArgs[2]
nimp=as.numeric(myArgs[3])
mit=as.numeric(myArgs[4])
minp=as.numeric(myArgs[5])
minc=as.numeric(myArgs[6])
data = read.csv(file, quote="\"")
#imputed <- mice(data, m=nimp, maxit=mit, meth=me, printFlag=FALSE)
# imputed <- mice(data, m=nimp, maxit=mit, pred=quickpred(data,minpuc=minp,mincor=minc), printFlag=FALSE)
imputed <- mice(data, m=nimp, maxit=mit, printFlag=FALSE)
#summary(imputed)
write.mice.imputation(mi.res=imputed, name=folder, dattype="csv")


