"""
Module To replicate experiments in section  5.2
"""
#######################################################################
## Imports
#######################################################################
import numpy as np
import OtherMethods as om
import SpectralMethod as sm
import RandomGenerator as rg
import matplotlib.pyplot as pl
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import time
#######################################################################
## Experiment
#######################################################################


#Set model dimension
n = 100
k = 5

#Initialize variables
wN = range(100, 1000,50)
L = 10
CLAccuracySVTD = np.zeros((L, len(wN)))
CLAccuracyTPM = np.zeros((L, len(wN)))
CLAccuracyEigen = np.zeros((L, len(wN)))
CLAccuracySVD = np.zeros((L, len(wN)))

ErrSVTD = np.zeros((L, len(wN)))
ErrTPM = np.zeros((L, len(wN)))
ErrEigen = np.zeros((L, len(wN)))
ErrSVD = np.zeros((L, len(wN)))

TimeSVTD = np.zeros((L,len(wN)))
TimeTPM = np.zeros((L, len(wN)))
TimeEigen = np.zeros((L,len(wN)))
TimeSVD = np.zeros((L,len(wN)))

NSteps = []
ErrC = []
ErrT = []
ErrFW = []

for i in range(L):
    nn=0
    for N in wN:
        # Generate Data
        X, M, omega, x = rg.generate_sample_SingleTopic(N, n, k, 20)
        wM2,wM3 = sm.RecoverMoments(omega, M)
        NSteps.append(N)

        # Retrieve empirical tensors
        print('.............Getting matrices')
        M1, M2, M3 = sm.RetrieveTensorsST(X)

        print('.............SVTD is working')
        t0 = time.time()
        # Parameters Learning
        MSVTD, omegaSVTD = sm.learn_LVM_Core(M1, M2, M3, k)
        TimeSVTD[i,nn] = time.time() - t0
        _,wM3h = sm.RecoverMoments(omegaSVTD, MSVTD)
        CL = sm.AssignClustersSingleTopic(MSVTD, omegaSVTD, X)
        #Calculates clustering accuracy and learning error
        CLAccuracySVTD[i, nn] = adjusted_rand_score(x, CL)
        ErrSVTD[i, nn] = np.linalg.norm(wM3h - wM3)

        ##
        print('.............Eigen is working')
        t0 = time.time()
        # Parameters Learning
        MEigen, omegaEigen = om.learn_LVM_AnanHMM12(M1, M2, M3, k)
        TimeEigen[i,nn] = time.time() - t0
        _,wM3h = sm.RecoverMoments(omegaEigen, MEigen)
        CL = sm.AssignClustersSingleTopic(MEigen, omegaEigen, X)
        #Calculates clustering accuracy and learning error
        CLAccuracyEigen[i, nn] =adjusted_rand_score(x, CL)
        ErrEigen[i, nn] = np.linalg.norm(wM3h - wM3)
        ##
        print('.............SVD is working')
        t0 = time.time()
        # Parameters Learning
        MSVD, omegaSVD = om.learn_LVM_AnanLDA12(M1, M2, M3, k)
        TimeSVD[i,nn] = time.time() - t0
        _,wM3h = sm.RecoverMoments(omegaSVD, MSVD)
        CL = sm.AssignClustersSingleTopic(MSVD, omegaSVD, X)
        #Calculates clustering accuracy and learning error
        CLAccuracySVD[i, nn] =adjusted_rand_score(x, CL)
        ErrSVD[i, nn] = np.linalg.norm(wM3h - wM3)
        #
        print('.............TPM is working')
        t0 = time.time()
        # Parameters Learning
        MTPM, omegaTPM = om.learn_LVM_Tensor14(M2, M3, k)
        TimeTPM[i, nn] = time.time() - t0
        _,wM3h = sm.RecoverMoments(omegaTPM, MTPM)
        CL = sm.AssignClustersSingleTopic(MTPM, omegaTPM, X)
        #Calculates clustering accuracy and learning error
        CLAccuracyTPM[i, nn] =adjusted_rand_score(x, CL)
        ErrTPM[i, nn] = np.linalg.norm(wM3h - wM3)
        print(N)
        nn+=1

#Remove the worst case example
for i in range(len(wN)):
    ErrTPM[np.argmax(ErrTPM[:, i]), i] = ErrTPM[np.argsort(ErrTPM[:, i])[::-1][1], i]
    ErrSVD[np.argmax(ErrSVD[:, i]), i] = ErrSVD[np.argsort(ErrSVD[:, i])[::-1][1], i]
    ErrEigen[np.argmax(ErrEigen[:, i]), i] = ErrEigen[np.argsort(ErrEigen[:, i])[::-1][1], i]
    ErrSVTD[np.argmax(ErrSVTD[:, i]), i] = ErrSVTD[np.argsort(ErrSVTD[:, i])[::-1][1], i]
    CLAccuracyTPM[np.argmax(CLAccuracyTPM[:, i]), i] = CLAccuracyTPM[np.argsort(CLAccuracyTPM[:, i])[::-1][1], i]
    CLAccuracySVD[np.argmax(CLAccuracySVD[:, i]), i] = CLAccuracySVD[np.argsort(CLAccuracySVD[:, i])[::-1][1], i]
    CLAccuracyEigen[np.argmax(CLAccuracyEigen[:, i]), i] = CLAccuracyEigen[np.argsort(CLAccuracyEigen[:, i])[::-1][1], i]
    CLAccuracySVTD[np.argmax(CLAccuracySVTD[:, i]), i] = CLAccuracySVTD[np.argsort(CLAccuracySVTD[:, i])[::-1][1], i]

# Plot data
f,a = pl.subplots(1,1, sharex=True)
cb = 'unit_points'
a.set_xlabel('N')
a.set_ylabel('Err')

sns.tsplot(data=ErrTPM, marker ='+', color ='green', time = wN)#, err_style = cb)
sns.tsplot(data=ErrSVD, marker ='*', color ='cyan', time = wN)#, err_style = cb)
sns.tsplot(data=ErrEigen, marker ='^', color ='red', time = wN)#, err_style = cb)
sns.tsplot(data=ErrSVTD, marker ='.', color ='blue', time = wN)#, err_style = cb)

f,a = pl.subplots(1,1, sharex=True)

a.set_xlabel('N')
a.set_ylabel('Adj. Rand. Index')

sns.tsplot(data=CLAccuracyTPM, marker ='+', color ='green', time = wN, condition='TPM')#, err_style = cb)
sns.tsplot(data=CLAccuracySVD, marker ='*', color ='cyan', time = wN, condition='SVD method')#, err_style = cb)
sns.tsplot(data=CLAccuracyEigen, marker ='^', color ='red', time = wN, condition='Eigen.')#, err_style = cb)
sns.tsplot(data=CLAccuracySVTD, marker ='.', color ='blue', time = wN, condition='SVTD')#, err_style = cb)


a.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pl.gcf().subplots_adjust(right=0.8)


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



