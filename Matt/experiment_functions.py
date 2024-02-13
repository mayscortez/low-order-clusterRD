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

def z_tilde(Z,Ni,beta):
    '''
    Returns |Si| vector of subset treatment indicators
        Z = treatment assignments: n x r
        Ni = neighborhood
        beta = model degree
    '''
    if Z.ndim == 1:
        return np.array([np.prod(np.take(Z,S)) for k in range(0,beta+1) for S in combinations(Ni,k)])
    else:
        return np.array([np.prod(Z[S,:], axis=0) for k in range(0,beta+1) for S in combinations(Ni,k)])
    
def outcomes(G, C, beta):
    ''' 
    Returns function that maps treatment assignment (n x r) to outcomes (n x r) 
        G = adjacency list representation of causal network
        C = effect coefficients
        beta = model degree
    '''
    return lambda Z: np.array([C[i].dot(z_tilde(Z,G[i],beta)) for i in range(len(Z))]) if Z.ndim == 1 else \
        np.stack([C[i].dot(z_tilde(Z,G[i],beta)) for i in range(Z.shape[0])], axis=1)

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

def staggered_rollout_bern(n,P,r=1):
    '''
        Returns treatment samples from Bernoulli staggered rollout
        n = number of individuals
        P = treatment probabilities for each time step: beta+1
        r = number of replications
    '''

    Z = np.zeros((len(P),n,r))
    U = np.random.rand(n,r)     # random values that determine when individual i starts being treated

    for t in range(len(P)):
        Z[t,:,:] = (U < P[t]) + 0

    return Z

def uncorrelated_bern(n,P,r=1):
    '''
        Returns treatment samples from independent multi-stage Bernoulli experiment
        n = number of individuals
        P = treatment probabilities for each time step: beta+1
        r = number of replications
    '''

    Z = np.zeros((len(P),n,r))

    for t in range(len(P)):
        Z[t,:,:] = (np.random.rand(n,r) < P[t]) + 0

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
        Z = treatment assignments: (beta+1) x n x r
        Y = potential outcomes function: {0,1}^n -> R^n
        P = treatment probabilities for each time step: beta+1
    '''
    T,n,r = Z.shape

    H = interp_coefficients(P)

    TTE_hat = np.zeros(r)
    for t in range(T):
        TTE_hat += H[t]*np.sum(Y(Z[t,:]),axis=1)
    
    return TTE_hat/n