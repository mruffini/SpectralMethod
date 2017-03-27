"""
Module containig implementations of alternative tensor decomposition methods.
"""
#######################################################################
## Imports
#######################################################################
import numpy as np
import sktensor as skt

#######################################################################
## Learning functions
#######################################################################


# A Method of Moments for Mixture Models and Hidden Markov Models
def learn_LVM_AnanHMM12(M1, M2, M3, k,Eta = None):
    """
    Algorithm A from  "A Method of Moments for Mixture Models and Hidden Markov Models"
    @param M1,M2,M3: the symmetric moments
    @param k: the number of latent states
    @param Eta: the randomization parameter; if None it is calculated in the function
    """

    d, _ = M2.shape

    U, s, V = np.linalg.svd(M2)
    U = U[:, 0:k]
    V = V[0:k,:]

    if Eta is None:
        Eta = np.random.uniform(0,1,d)
        Eta = Eta/np.linalg.norm(Eta)

    Tr = np.einsum('ijk,k->ij', M3, Eta)

    _,Xs = np.linalg.eig((U.T.dot(Tr).dot(V.T)).dot(np.linalg.pinv((U.T.dot(M2).dot(V.T)))))

    M = U.dot(Xs).dot(np.diag(np.max(np.sign(U.dot(Xs)),0)))
    M = M/np.sum(M,0)

    x = np.linalg.lstsq(M, M1)
    omega = x[0]
    omega = omega / sum(omega)

    return M, omega

# A spectral algorithm for latent dirichlet allocation
def learn_LVM_AnanLDA12(M1, M2, M3, k, Eta = None):
    """
    Algorithm 1 from  "A spectral algorithm for latent dirichlet allocation"
    @param M1,M2,M3: the symmetric moments
    @param k: the number of latent states
    @param Eta: the randomization parameter; if None it is calculated in the function
    """

    d, _ = M2.shape
    U, s, _ = np.linalg.svd(M2)
    W = np.linalg.pinv(U[:,:k].dot(np.diag(np.sqrt(s[:k])))).T

    if Eta is None:
        Eta = np.random.normal(0,1,k)
        Eta = Eta/np.linalg.norm(Eta)


    Tr = np.einsum('ijk,k->ij', M3, W.dot(Eta))
    Xs,_,_ = np.linalg.svd(W.T.dot(Tr).dot(W))

    M = np.linalg.pinv(W).T.dot(Xs).dot(np.diag(np.max(np.sign(np.linalg.pinv(W.T).dot(Xs)),0)))
    x = np.linalg.lstsq(M, M1)
    omega = x[0] ** 2
    omega = omega / sum(omega)

    M = M.dot(np.linalg.pinv(np.diag(np.sqrt(omega))))
    M = M/np.sum(M,0)

    return M, omega

# Tensor Decompositions for Learning Latent Variable Models
def learn_LVM_Tensor14(M2, M3, k, L=25, N=20):
    """
    Theorem 4.3 from  "Tensor Decompositions for Learning Latent Variable Models"
    @param M2,M3: the symmetric moments
    @param k: the number of latent states
    @param L,N: number of iterations
    """
    d,ds = M2.shape
    u, s, _ = np.linalg.svd(M2)
    u = u[:,:k]
    s = s[:k]

    W = u.dot(np.diag(1/np.sqrt(s)))

    T = skt.dtensor(M3).ttm([W.T, W.T, W.T])

    Thetas = []
    Lambdas = []

    wT = T.copy()
    for i in range(k):
        theta, Lambda, wT = RobustTPM(wT.copy(),k,L,N)
        Thetas.append(theta)
        Lambdas.append(Lambda)

    Thetas = np.array(Thetas)
    Lambdas = np.array(Lambdas)

    B = np.linalg.pinv(W.T)
    pi = 1 / Lambdas ** 2

    M = np.zeros((d, k))

    for j in range(k):
        M[:, j] = Lambdas[j] * B.dot(Thetas[j, :].reshape(k,1)).reshape(d)
    M = M/np.sum(M,0)

    omega = pi
    omega = omega / sum(omega)

    return M, omega

def RobustTPM(T,k,L=25, N=20):
    """
    Algorithm 1 from   "Tensor Decompositions for Learning Latent Variable Models"
    @param T: symmetric tensor
    @param k: the number of latent states
    @param L,N: number of iterations
    """
    Thetas = []

    for tau in range(L):
        Theta = np.random.randn(1,k)
        Theta = Theta / np.linalg.norm(Theta)

        for t in range(N):
            Theta = T.ttm([Theta, Theta], [1, 2]).reshape(1, k)
            Theta = Theta / np.linalg.norm(Theta)

        Thetas.append(Theta)

    ThetaFinal_idx = np.argmax([T.ttm([theta, theta, theta], [0, 1, 2]) for theta in Thetas])

    Theta = Thetas[ThetaFinal_idx]

    for t in range(N):
        Theta = T.ttm([Theta, Theta], [1, 2]).reshape(1, k)
        Theta = Theta / np.linalg.norm(Theta)

    Lambda = T.ttm([Theta, Theta, Theta], [0, 1, 2]).squeeze()

    return Theta, Lambda, T - skt.ktensor([Theta.T, Theta.T, Theta.T]).totensor() * Lambda
