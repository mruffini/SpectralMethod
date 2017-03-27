"""
Module To replicate experiments in section  6.2.1
"""
#######################################################################
## Imports
#######################################################################
import numpy as np
import OtherMethods as om
import SpectralMethod as sm
import RandomGenerator as rg
import matplotlib.pyplot as pl
import time
#######################################################################
## Experiment
#######################################################################

#Set model dimension
n = 99
k = 5

#Initialize variables
ErrSVTD = []
ErrTPM = []
ErrEigen = []
ErrSVD = []
TimeSVTD = []
TimeTPM = []
TimeEigen = []
TimeSVD = []
NSteps = []
ErrC = []
ErrT = []
ErrFW = []
MinDiff = []
for N in range(50, 1000,50):
    NSteps.append(N)
    print "Iteration num: %s" % (N)

    print('.............Getting matrices')
    X, M, omega = rg.generate_sample_SingleTopic(N, n, k,4)
    MinDiff.append(np.diff(np.sort(M,1),1).min(1).max())
    M1, M2, M3 = sm.RetrieveTensorsST(X)
    wM2 = M.dot(np.diag(omega)).dot(M.T)

    print('.............SVTD is working')
    t0 = time.time()
    MLSVTD, POFSVTD = sm.learn_LVM_Core(M1, M2, M3, k)
    TimeSVTD.append(time.time()-t0)
    ErrSVTD.append(np.linalg.norm(MLSVTD.dot(np.diag(POFSVTD)).dot(MLSVTD.T) - wM2))
    ##
    print('.............Eigen is working')
    t0 = time.time()
    MLEigen, POFEigen = om.learn_LVM_AnanHMM12(M1, M2, M3, k)
    TimeEigen.append(time.time() - t0)
    ErrEigen.append(np.linalg.norm(MLEigen.dot(np.diag(POFEigen)).dot(MLEigen.T) - wM2))
    ## #
    print('.............SVD is working')
    t0 = time.time()
    MLSVD, POFSVD = om.learn_LVM_AnanLDA12(M1, M2, M3, k)
    TimeSVD.append(time.time()-t0)
    ErrSVD.append(np.linalg.norm(MLSVD.dot(np.diag(POFSVD)).dot(MLSVD.T) - wM2))
    #
    print('.............TPM is working')
    t0 = time.time()
    MLTPM, POFTPM = om.learn_LVM_Tensor14(M2, M3, k, 20, 50)
    TimeTPM.append(time.time()-t0)
    ErrTPM.append(np.linalg.norm(MLTPM.dot(np.diag(POFTPM)).dot(MLTPM.T) - wM2))




TimeTPM = np.array(TimeTPM)
TimeSVTD = np.array(TimeSVTD)
TimeEigen = np.array(TimeEigen)
TimeSVD = np.array(TimeSVD)

#Plot Charts
pl.plot(NSteps,ErrSVTD, label ='SVTD', lw = 2, ls='-')
pl.plot(NSteps, ErrTPM, label ='Tensor power method')
pl.plot(NSteps, ErrEigen, label ='Eigendecomposition method')
pl.plot(NSteps, ErrSVD, label ='SVD method')
pl.xlabel("N")
pl.ylabel("Err")
pl.legend()

f, ax1 = pl.subplots(1, figsize=(10,5))
bar_width = 0.75
bar_l = [i+1 for i in range(4)]
tick_pos = [i+(bar_width/2) for i in bar_l]
ax1.bar(bar_l,
        [TimeSVTD.mean(), TimeEigen.mean(), TimeSVD.mean(), TimeTPM.mean()],
        width=bar_width,
        label='Time',
        alpha=0.5,
        color='#F4561D')

pl.xticks(tick_pos, ['SVTD','Eigen.','SVD','TPM'])
ax1.set_ylabel("Seconds")
pl.legend(loc='upper left')
pl.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
