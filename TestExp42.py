"""
Module To replicate experiments in section  4.2
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
import itertools
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
CLAccuracyCP = np.zeros((L, len(wN)))

ErrSVTD = np.zeros((L, len(wN)))
ErrTPM = np.zeros((L, len(wN)))
ErrEigen = np.zeros((L, len(wN)))
ErrSVD = np.zeros((L, len(wN)))
ErrCP = np.zeros((L, len(wN)))

TimeSVTD = np.zeros((L,len(wN)))
TimeTPM = np.zeros((L, len(wN)))
TimeEigen = np.zeros((L,len(wN)))
TimeSVD = np.zeros((L,len(wN)))
TimeCP = np.zeros((L,len(wN)))

NSteps = []
ErrC = []
ErrT = []
ErrFW = []

perm = list(itertools.permutations(range(k)))

def get_error(true_M,M):
    M[M < 0.00000001] = 0.00000001
    M[M > 1] = 1 - 0.00000001
    M = M / M.sum(0)
    M = M/M.sum(0)
    err = np.inf
    for p in perm:
        wM = M[:,list(p)]
        werr = np.linalg.norm(wM-true_M,2)
        if werr<err:
            err = werr
    return err

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
        CL = sm.AssignClustersSingleTopic(MSVTD, omegaSVTD, X)
        #Calculates clustering accuracy and learning error
        CLAccuracySVTD[i, nn] = adjusted_rand_score(x, CL)
        ErrSVTD[i, nn] = get_error(M,MSVTD)

        ##
        print('.............Eigen is working')
        t0 = time.time()
        # Parameters Learning
        MEigen, omegaEigen = om.learn_LVM_AnanHMM12(M1, M2, M3, k)
        TimeEigen[i,nn] = time.time() - t0
        CL = sm.AssignClustersSingleTopic(MEigen, omegaEigen, X)
        #Calculates clustering accuracy and learning error
        CLAccuracyEigen[i, nn] =adjusted_rand_score(x, CL)
        ErrEigen[i, nn] =  get_error(M,MEigen)
        ##
        print('.............SVD is working')
        t0 = time.time()
        # Parameters Learning
        MSVD, omegaSVD = om.learn_LVM_AnanLDA12(M1, M2, M3, k)
        TimeSVD[i,nn] = time.time() - t0
        CL = sm.AssignClustersSingleTopic(MSVD, omegaSVD, X)
        #Calculates clustering accuracy and learning error
        CLAccuracySVD[i, nn] =adjusted_rand_score(x, CL)
        ErrSVD[i, nn] =  get_error(M,MSVD)
        #
        print('.............TPM is working')
        t0 = time.time()
        # Parameters Learning
        MTPM, omegaTPM = om.learn_LVM_Tensor14(M2, M3, k)
        TimeTPM[i, nn] = time.time() - t0
        CL = sm.AssignClustersSingleTopic(MTPM, omegaTPM, X)
        #Calculates clustering accuracy and learning error
        CLAccuracyTPM[i, nn] =adjusted_rand_score(x, CL)
        ErrTPM[i, nn] =  get_error(M,MTPM)
        print('.............CPD is working')
        t0 = time.time()
        # Parameters Learning
        MCP, omegaCP = om.learn_LVM_CPD(M1,M2, M3, k)
        TimeCP[i, nn] = time.time() - t0
        CL = sm.AssignClustersSingleTopic(MCP, omegaCP, X)
        #Calculates clustering accuracy and learning error
        CLAccuracyCP[i, nn] =adjusted_rand_score(x, CL)
        ErrCP[i, nn] =  get_error(M,MCP)
        print(N)
        nn+=1

#Remove the worst case example
for i in range(len(wN)):
    ErrTPM[np.argmax(ErrTPM[:, i]), i] = ErrTPM[np.argsort(ErrTPM[:, i])[::-1][1], i]
    ErrSVD[np.argmax(ErrSVD[:, i]), i] = ErrSVD[np.argsort(ErrSVD[:, i])[::-1][1], i]
    ErrEigen[np.argmax(ErrEigen[:, i]), i] = ErrEigen[np.argsort(ErrEigen[:, i])[::-1][1], i]
    ErrSVTD[np.argmax(ErrSVTD[:, i]), i] = ErrSVTD[np.argsort(ErrSVTD[:, i])[::-1][1], i]
    ErrCP[np.argmax(ErrCP[:, i]), i] = ErrCP[np.argsort(ErrCP[:, i])[::-1][1], i]
    CLAccuracyTPM[np.argmax(CLAccuracyTPM[:, i]), i] = CLAccuracyTPM[np.argsort(CLAccuracyTPM[:, i])[::-1][1], i]
    CLAccuracySVD[np.argmax(CLAccuracySVD[:, i]), i] = CLAccuracySVD[np.argsort(CLAccuracySVD[:, i])[::-1][1], i]
    CLAccuracyEigen[np.argmax(CLAccuracyEigen[:, i]), i] = CLAccuracyEigen[np.argsort(CLAccuracyEigen[:, i])[::-1][1], i]
    CLAccuracySVTD[np.argmax(CLAccuracySVTD[:, i]), i] = CLAccuracySVTD[np.argsort(CLAccuracySVTD[:, i])[::-1][1], i]
    CLAccuracyCP[np.argmax(CLAccuracyCP[:, i]), i] = CLAccuracyCP[np.argsort(CLAccuracyCP[:, i])[::-1][1], i]

# Plot data
f,a = pl.subplots(1,1, sharex=True)
cb = 'unit_points'
a.set_xlabel('N',fontsize = 20)
a.set_ylabel('Err',fontsize = 20)
pl.yticks( fontsize = 20)
pl.xticks( fontsize = 20)
sns.tsplot(data=ErrCP, marker ='o', color ='m', time = wN)#, err_style = cb)
sns.tsplot(data=ErrTPM, marker ='+', color ='green', time = wN)#, err_style = cb)
sns.tsplot(data=ErrSVD, marker ='*', color ='cyan', time = wN)#, err_style = cb)
sns.tsplot(data=ErrEigen, marker ='^', color ='red', time = wN)#, err_style = cb)
sns.tsplot(data=ErrSVTD, marker ='.', color ='blue', time = wN)#, err_style = cb)
pl.gcf().subplots_adjust(bottom=0.15)

f,a = pl.subplots(1,1, sharex=True)

a.set_xlabel('N', fontsize = 20)
a.set_ylabel('Adj. Rand. Index',fontsize = 20)

sns.tsplot(data=CLAccuracyCP, marker ='o', color ='m', time = wN, condition='ALS')#, err_style = cb)
sns.tsplot(data=CLAccuracyTPM, marker ='+', color ='green', time = wN, condition='TPM')#, err_style = cb)
sns.tsplot(data=CLAccuracySVD, marker ='*', color ='cyan', time = wN, condition='SVD method')#, err_style = cb)
sns.tsplot(data=CLAccuracyEigen, marker ='^', color ='red', time = wN, condition='Eigen.')#, err_style = cb)
sns.tsplot(data=CLAccuracySVTD, marker ='.', color ='blue', time = wN, condition='SVTD')#, err_style = cb)


a.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 20)
pl.yticks( fontsize = 20)
pl.xticks( fontsize = 20)
a.set_ylim([0,.7])

pl.gcf().subplots_adjust(right=0.7)
pl.gcf().subplots_adjust(bottom=0.15)


f, ax1 = pl.subplots(1, figsize=(10,5))
bar_width = 0.75
bar_l = [i+1 for i in range(5)]
tick_pos = [i+(bar_width/2) for i in bar_l]
ax1.bar(bar_l,
        [TimeSVTD.mean(), TimeEigen.mean(), TimeSVD.mean(), TimeTPM.mean(), TimeCP.mean()],
        width=bar_width,
        label='Time',
        alpha=0.5,
        )

pl.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
ax1.set_ylabel("Seconds",fontsize = 20)
pl.yticks( fontsize = 20)
pl.xticks(tick_pos, ['SVTD','Eigen.','SVD','TPM', 'ALS'], fontsize = 20)

pl.legend(loc='upper left', fontsize = 20)
pl.ylabel("Seconds",fontsize = 20)


