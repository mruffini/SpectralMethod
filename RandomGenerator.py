"""
Module to generate synthetic data.
"""

#######################################################################
## Imports
#######################################################################
import numpy as np
#######################################################################
## Data generators
#######################################################################

def generate_sample_SingleTopic(N, n, k, c=None):

    """
    Generates a sample of N synthetic bag-of-words documents distributed
    as a Single Topic Model.
    Return:
    the generated samples X, with N rows an n columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    the topic-word probability matrix M, with n rows an k columns;
        at position (i,j) we have the probability of the word i under topic k.
    the topic probability array omega, with k entries.
        at position (i) we have the probability of drawing topic i.
    the true topic of each text
    @param N: The number of synthetic documents to be generated
    @param n: the size of the vocabulary
    @param k: the number of hidden topics
    @param c: the number of words appearing in each document. If None, documents have random lengths
    """

    omega = np.random.uniform(0, 1, k)
    omega = omega / np.sum(omega)

    M = np.random.uniform(0, 1, [n, k])*(n*k)+20
    M = M / np.sum(M, 0)

    #Assign the hidden topic for each document
    x = np.random.multinomial(1, omega, N)
    x = np.argmax(x, 1)

    X = np.zeros((N,n))
    #Generates the documents
    for i in range(k):
        wN = int(sum(x == i))
        if c is None:
            X[x == i, :] = np.array([np.random.multinomial(np.random.randint(3,100), M[:, i],1) for j in range(wN)]).reshape(wN,n)
        else:
            X[x == i, :] = np.array([np.random.multinomial(c, M[:, i],1) for j in range(wN)]).reshape(wN,n)

    return X.astype(float), M, omega, x

def generate_sample_LDA(N, n, k, c, Alpha0=2):
    """
    Generates a sample of N synthetic bag-of-words documents distributed
    as in Latent Dirichlet Allocation.
    Return:
    the generated samples X, with N rows an n columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    the topic-word probability matrix M, with n rows an k columns;
        at position (i,j) we have the probability of the word i under topic k.
    the hyperparameter Alpha of the Dirichlet distribution.
    @param N: The number of synthetic documents to be generated
    @param n: the size of the vocabulary
    @param k: the number of hidden topics
    @param c: the number of words appearing in each document
    @param Alpha0: the sum of the hyperparameter of the Dirichlet distribution
    """

    Alpha = np.random.uniform(0, 1, k)
    Alpha = Alpha / np.sum(Alpha)*Alpha0

    M = np.random.uniform(0, 1, [n, k])*(n*k)+20
    M = M / np.sum(M, 0)

    #Assign the mixture of topics for each document
    h = np.random.dirichlet(Alpha, N)

    #Generates the topics for the individual words
    x = np.zeros((N,c))
    for i in range(N):
        x[i] = np.argmax(np.random.multinomial(1,h[i,:],c),1)

    #Generates the individual words
    Xr = np.zeros((N,c))
    for i in range(k):
        Xr[x==i] = np.argmax(np.random.multinomial(1, M[:, i],int(np.sum(x==i) )),1)

    #Generates the documents
    X = np.zeros((N,n))
    for i in range(n):
        X[:,i] = np.sum(Xr==i,1)

    return X, M, Alpha
