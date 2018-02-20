"""
Module To replicate experiments in section  4.1
"""
#######################################################################
## Imports
#######################################################################
import matplotlib.pyplot as pl
from SpectralMethod import *
from RandomGenerator import *
import seaborn as sns
#######################################################################
## Experiment
#######################################################################

#Set model dimension
N = 1000
n = 100
k = 5

#Initialize variables
Err3_zou=[]
Err3=[]
Err2_zou=[]
Err2=[]
xax = []
for N in range(100,1000,50):
    print "Iteration num: %s" % (N)
    X, M, omega,_ = generate_sample_SingleTopic(N,n,k)
    # Retrieve empirical tensors
    _, M2_prop, M3_prop = RetrieveTensorsST(X)
    M2_zou, M3_zou = RetrieveTensorsST_Zou(X)
    M2, M3 = RecoverMoments(omega, M)
    # calculates errors
    Err2_zou.append(np.linalg.norm(M2-M2_zou))
    Err2.append(np.linalg.norm(M2-M2_prop))
    Err3_zou.append(np.linalg.norm(M3-M3_zou))
    Err3.append(np.linalg.norm(M3-M3_prop))
    xax.append(N)


# plot results
f, ax = pl.subplots(1, 2)
ax[0].plot(xax,Err2_zou,label='Zou et al.')
ax[0].plot(xax,Err2,label='Thm 2.1')
ax[0].set_title('Error on M2')
ax[0].set_xlabel("N")
ax[0].set_ylabel("Err")
ax[1].plot(xax,Err3_zou,label='Zou et al.')
ax[1].plot(xax,Err3,label='Thm 2.1')
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].set_title('Error on M3')
ax[1].set_xlabel("N")

pl.gcf().subplots_adjust(bottom=0.2)
