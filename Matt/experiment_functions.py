import numpy as np
from math import comb
from itertools import combinations, chain

######## Graph ########

def sbm(n,k,pii,pij):
    ''' 
    Returns a graph (adjacency list of in-neighbors) sampled from a stochastic block model
        n = number of vertices
        k = number of communities
        pii = edge probability within community
        pij = edge probability across communities
    '''

    c = n//k  # community size

    A = (np.random.rand(n,n) < pij) + 0
    for i in range(k):
        A[i*c:(i+1)*c,i*c:(i+1)*c] = (np.random.rand(c,c) < pii) + 0

    A[range(n),range(n)] = 1   # everyone is affected by their own treatment

    return list([np.nonzero(A[i,:])[0] for i in range(n)])

######## Coefficients ########

def random_weights_degree_scaled(G, beta):
    ''' 
    Returns weights randomly sampled from [0,10], then rescaled by (1/2)^|S|
        G = adjacency list representation of causal network
        beta = model degree
    '''
    C = []

    for Ni in G:
        Ci = []
        d = len(Ni)

        for k in range(beta+1):
            Ci += list(np.random.rand(comb(d,k))*10/(3**k))
        C.append(np.array(Ci))

    return C

######## Potential Outcomes ########

def z_tilde(z,Ni,beta):
    '''
    Returns |Si| vector of subset treatment indicators
        z = treatment assignments
        Ni = neighborhood
        beta = model degree
    '''
    zn = np.take(z,Ni) # treatment assignments of neighbors
    return np.array([np.prod(S) for k in range(0,beta+1) for S in combinations(zn,k)])
    
def outcomes(G, C, beta):
    ''' 
    Returns function that maps treatment assignment (n-vector) to outcomes (n-vector) 
        G = adjacency list representation of causal network
        C = effect coefficients
        beta = model degree
    '''
    return lambda z: np.array([z_tilde(z,G[i],beta).dot(C[i]) for i in range(len(z))])

def true_poly(G,C,beta):
    ''' 
    Returns coefficients of polynomial E[1/n sum Yi(z)] 
        G = adjacency list representation of causal network
        C = effect coefficients
        beta = model degree
    '''
    n = len(G)
    
    F = np.zeros(beta+1)
    for i in range(n):
        Ni = G[i]
        d = len(Ni)
        for (j,S) in enumerate(chain.from_iterable(combinations(Ni, k) for k in range(beta+1))):
            F[len(S)] += C[i][j]
    
    return F/n

######## Treatment Assignments ########

def staggered_rollout_bern(n, P):
    '''
        Returns treatment samples from Bernoulli staggered rollout
        n = number of individuals
        P = treatment probabilities for each time step: beta+1
    '''

    Z = np.zeros((len(P),n))
    U = np.random.rand(n)   # random values that determine when individual i starts being treated

    for t in range(len(P)):
        Z[t,:] = (U < P[t]) + 0

    return Z

######## Estimator ########

def interp_coefficients(P):
    '''
    Returns coefficients h_t = l_t,P(1) - l_t,P(0) for pi estimator
        P = treatment probabilities for each time step: beta+1
    '''
    T = len(P)

    H = np.zeros(T)

    for t in range(T):
        denom = np.prod([(P[t] - P[s]) for s in range(T) if s != t])
        H[t] = np.prod([(1 - P[s]) for s in range(T) if s != t]) 
        H[t] -= np.prod([(-P[s]) for s in range(T) if s != t])
        H[t] /= denom

    return H

def pi_estimate_tte(Z,Y,P):
    '''
    Returns TTE estimate from polynomial interpolation
        Z = treatment assignments: (beta+1) x n
        Y = potential outcomes function: {0,1}^n -> R^n
        P = treatment probabilities for each time step: beta+1
    '''
    T,n = Z.shape

    H = interp_coefficients(P)

    TTE_hat = 0
    for t in range(T):
        TTE_hat += H[t]*np.sum(Y(Z[t,:]))
    
    return TTE_hat/n