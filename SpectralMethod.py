#######################################################################
## Imports
#######################################################################
import numpy as np
import scipy as sp
import sktensor as skt

#######################################################################
## Learning functions
#######################################################################


def RecoverMoments(omega, M):
    """
    Recovers the theoretical moment matrix M2 and the tensors M3 from the conditional expectations
    and the mixing weights
    @param omega: the mixing weights
    @param M: the conditional expectations matrix
    """
    M2 = M.dot(np.diag(omega)).dot(M.T)
    M3 = skt.ktensor([M,M,M], omega).totensor()

    return M2,M3

def RetrieveTensorsST(X):
    """
    Returns a the three tensors M1, M2 and M3 to be used
    to learn the Single Topic Model, as in theorem 2.1
    @param X: a bag-of-words documents distributed
        as a Single Topic Model, with N rows an n columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    """

    (N, n) = np.shape(X)

    M1 = np.sum(X,0)/np.sum(X)

    W = X - 1
    W[W < 0] = 0
    W2 = X - 2
    W2[W2 < 0] = 0
    Num = X * W
    Den = np.sum(X, 1)
    wDen = Den - 1
    wDen[wDen < 0] = 0
    wwDen = Den - 2
    wwDen[wwDen < 0] = 0

    Den1 = sum(Den * wDen)
    Den2 = sum(Den * wDen * wwDen)

    Diag = np.sum(Num, 0) / Den1

    M2 = sp.transpose(X).dot(X) / Den1
    M2[range(n), range(n)] = Diag

    M3 = np.zeros((n, n, n))
    for j in range(n):
        Y = X[:, j].reshape((N, 1))
        Num = X * Y * W
        Diag = np.sum(Num, 0) / Den2
        wM3 = (Y * X).T.dot(X) / Den2
        wM3[range(n), range(n)] = Diag
        rr = np.sum(Y * W[:, j].reshape((N, 1)) * X,0) / Den2
        wM3[j, :] = rr
        wM3[:, j] = rr
        wM3[j, j] = np.sum(Y * W[:, j].reshape((N, 1)) * W2[:, j].reshape((N, 1))) / Den2
        M3[j] = wM3

    return M1, M2, M3

def RetrieveTensorsST_Zou(X):
    """
    Returns a the three tensors M1, M2 and M3 to be used
    to learn the Single Topic Model, Zou et al 2013
    @param X: a bag-of-words documents distributed
        as a Single Topic Model, with N rows an n columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    """

    (N, n) = np.shape(X)

    W = X - 1
    W[W < 0] = 0
    W2 = X - 2
    W2[W2 < 0] = 0
    Num = X * W

    Dn = (np.sum(X,1)*(np.sum(X,1)-1)*(np.sum(X,1)-2)).reshape(N,1)

    Cn = (np.sum(X,1)*(np.sum(X,1)-1)).reshape(N,1)
    Diag = np.mean(Num/Cn, 0)

    M2 = np.transpose(X/Cn).dot(X)/N
    M2[range(n), range(n)] = Diag

    M3 = np.zeros((n, n, n))
    for j in range(n):
        Y = X[:, j].reshape((N, 1))
        Num = X * Y * W
        Diag = np.mean(Num/Dn, 0)
        wM3 = (Y * X/Dn).T.dot(X)/N

        wM3[range(n), range(n)] = Diag
        rr = np.mean(Y * W[:, j].reshape((N, 1)) * X/Dn, 0)
        wM3[j, :] = rr
        wM3[:, j] = rr
        wM3[j, j] = np.mean(Y * W[:, j].reshape((N, 1)) * W2[:, j].reshape((N, 1))/Dn)
        M3[j] = wM3

    return M2, M3

def RetrieveTensorsLDA(X,Alpha0):
    """
    Returns a the three tensors M1, M2a and M3a to be used
    to learn the Latent Dirichlet Allocation, as in theorem 3.1
    @param X: a bag-of-words documents distributed
        as in Latent Dirichlet Allocation, with N rows an n columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    @param Alpha0: the sum of the hyperparameter Alpha of the Dirichlet distribution.
    """

    N, n = np.shape(X)

    M1, M2, M3 = RetrieveTensorsST(X)

    M2a = M2 - Alpha0 / (1 + Alpha0) * M1.reshape(n, 1).dot(M1.reshape(1, n))
    wM1 = 2 * Alpha0 ** 2 / ((Alpha0 + 1) * (Alpha0 + 2)) * M1[:,None,None] * M1[:,None] * M1
    wM1_0 = Alpha0 / (Alpha0 + 2) *M2[:,:,None]*M1
    wM1_1 = Alpha0 / (Alpha0 + 2) * M2[:,None,:]*M1[None,:,None]
    wM1_2 = Alpha0 / (Alpha0 + 2) *M1[:,None,None]*M2
    M3a = M3 - (wM1_0 + wM1_1 + wM1_2) + wM1

    return M1, M2a, M3a

def learn_LVM_Core(M1, M2, M3, k):

    """
    Implementation of Algorithm 2 to learn the Sinlge topic model from a sample.
    Returns:
    the topic-word probability matrix M, with n rows an k columns;
        at position (i,j) we have the probability of the word i under topic k.
    the topic probability array omega, with k entries.
        at position (i) we have the probability of drawing topic i.
    @params M1, M2, M3: to be used to learn the Single Topic Model,
        from in theorem 2.1 (retrieved from RetrieveTensorsST)
    """
    n, col = M2.shape
    #Step 1
    u,s,v = np.linalg.svd(M2)
    #Step 2
    E = u[:, 0:k].dot((np.diag(np.sqrt(s[0:k]))))

    HMin = 0
    H = np.zeros([k,k,n])
    M = np.zeros([n,k])

    #We select the feature with the most different singular vectors
    for r in range(0,n):
        # Step 3
        M3r = np.delete(M3[:,:,r], r, 0)
        M3r = np.delete(M3r, r, 1)
        # Step 4
        Er = np.delete(E,r,0)
        pEr = np.linalg.pinv(Er)
        H[:,:,r] =  pEr.dot(M3r).dot(np.transpose(pEr))
        t = H[:,:,r]
        Or,s,v = np.linalg.svd(t)
        if np.min(-np.diff(s))>HMin:
            HMin = np.min(-np.diff(s))
            # Step 5
            O = Or

    #Step 6
    for r in range(0, n):
        # Step 8
        mur = np.diag(np.transpose(O).dot(H[:, :, r]).dot(O))
        M[r,:] = mur

    M = M/np.sum(M,0)
    #Step 10

    x = np.linalg.lstsq(M, M1)
    omega = x[0]
    omega = omega / sum(omega)

    return M, omega

def learn_LVM_Core_LDA(M1a, M2a, M3a, Alpha0, k):

    """
    Adaptation of Algorithm 1 to learn Latent Dirichlet Allocation from a sample.
    Returns:
    the topic-word probability matrix M, with n rows an k columns;
        at position (i,j) we have the probability of the word i under topic k.
    the hyperparameter Alpha of the Dirichlet distribution.
    @params M1a, M2a, M3a: to be used to learn the Single Topic Model,
        from in theorem 2.1 (retrieved from RetrieveTensorsST)
    @params Alpha0: the sum of the hyperparameter Alpha of the Dirichlet distribution.
    """

    M, omega = learn_LVM_Core(M1a, M2a, M3a, k)

    M = M*(Alpha0+2)/2
    M = M/np.sum(M,0)
    Alpha = np.linalg.pinv(M).dot(M1a) * Alpha0

    return M, Alpha


