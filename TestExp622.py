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
#######################################################################
## Experiment
#######################################################################

#Set model dimension
n = 100
k = 5

#Initialize variables
ErrSVTD = []
ErrTPM = []
ErrEigen = []
ErrSVD = []
NSteps = []
ErrC = []
ErrT = []
ErrFW = []
MLTPM_old = None
MLSVTD_Old = None
MLEigen_Old = None
MLSVD_Old = None

#Initialize the full data sample
X_T, M, omega = rg.generate_sample_SingleTopic(1000, n, k, 20)

#Fix once for all the randomization factors
Eta = np.random.uniform(0, 1, n)
Eta = Eta / np.linalg.norm(Eta)

Etak = np.random.uniform(0, 1, k)
Etak = Etak / np.linalg.norm(Etak)

for N in range(50, 200,1):
    # Append a single sample to the data
    X = X_T[:N,:]
    NSteps.append(N)
    print "Iteration num: %s" % (N)
    print('.............Getting matrices')
    M1, M2, M3 = sm.RetrieveTensorsST(X)

    print('.............SVTD is working')
    MLSVTD, POFw = sm.learn_LVM_Core(M1, M2, M3, k)
    MLSVTD = MLSVTD[:, MLSVTD[0].argsort()]
    if not MLSVTD_Old is None: ErrSVTD.append(np.linalg.norm(MLSVTD - MLSVTD_Old))
    MLSVTD_Old = MLSVTD
    ##
    print('.............Eigen is working')
    MLEigen, POFEigen = om.learn_LVM_AnanHMM12(M1, M2, M3, k,Eta)
    MLEigen = MLEigen[:, MLEigen[0].argsort()]
    if not MLEigen_Old is None: ErrEigen.append(np.linalg.norm(MLEigen-MLEigen_Old))
    MLEigen_Old = MLEigen
    ## #
    print('.............SVD is working')
    MLSVD, POFhmm = om.learn_LVM_AnanLDA12(M1, M2, M3, k,Etak)
    MLSVD = MLSVD[:, MLSVD[0].argsort()]
    if not MLSVD_Old is None: ErrSVD.append(np.linalg.norm(MLSVD_Old - MLSVD))
    MLSVD_Old = MLSVD
    #
    print('.............TPM is working')
    MLTPM, POFTPM = om.learn_LVM_Tensor14(M2, M3, k)
    MLTPM = MLTPM[:, MLTPM[0].argsort()]
    if not MLTPM_old is None: ErrTPM.append(np.linalg.norm(MLTPM-MLTPM_old))
    MLTPM_old = MLTPM


ErrSVD = np.array(ErrSVD)
ErrEigen = np.array(ErrEigen)

ErrEigen[ErrEigen>1] = .1
ErrSVD[ErrSVD>1] = .1

pl.plot(NSteps[1:],ErrTPM,'+', label = 'Tensor power method', color = 'green')
pl.plot(NSteps[1:],ErrSVD,'*', label = 'SVD method', color = 'cyan')
pl.plot(NSteps[1:],ErrEigen,'^', label = 'Eigendecomposition method', color = 'red')
pl.plot(NSteps[1:],ErrSVTD,'.', label = 'SVTD', lw = 2, ls='.', color = 'blue')
pl.xlabel("N")
pl.ylabel("Var")
pl.legend()
