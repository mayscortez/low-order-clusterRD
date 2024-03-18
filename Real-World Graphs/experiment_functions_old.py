import numpy as np
from itertools import combinations
from math import comb
from scipy.special import binom

######## Effect Coefficients ########

def homophily_effects(G):
    '''
    Returns vectors of (normalized) homophily effects as described in the Ugander/Yin paper
        G = adjacency list representation of graph
    '''

    n = len(G)
    degrees = [len(Ni) for Ni in G]

    normalized_laplacian = np.zeros((n,n))
    for i,Ni in enumerate(G):
        normalized_laplacian[i,Ni] = -1/degrees[i]
    normalized_laplacian[range(n),range(n)] = 1
    
    eigvals, eigvecs = np.linalg.eig(normalized_laplacian)
    fiedler_index = np.where(eigvals.real == np.sort(eigvals.real)[1])[0][0]
    h = eigvecs[:,fiedler_index].real

    return h/(max(abs(max(h)),abs(min(h))))

def modified_ugander_yin_weights(G,h,beta):
    '''
    Returns vectors of coefficients c_i,S for each individual i and neighborhood subset S 
    Coefficients are given by a modification of Ugander/Yin's model to incorporate non-uniform and higher-order neighbor effects
        G = adjacency list representation of graph
        h = vector of (scaled) homophily effects
        beta = model degree
    '''

    # parameters 
    a = 1                                         # baseline effect
    sigma = 0.1                                   # magnitude of random perturbation
    delta = 1                                     # magnitude of direct effect
    gamma = [0.5**(k-1) for k in range(beta+1)]   # magnitude of subset treatment effects

    n = len(G)
    degrees = [len(Ni) for Ni in G]
    dbar = sum(degrees)/n

    baseline = ( a + h + sigma * np.random.normal(size=n) ) * degrees/dbar

    C = []

    for i in range(n):
        Ci = [baseline[i]]
        for k in range(1,beta+1):
            for S in combinations(G[i],k):
                Ci.append(baseline[i] * (gamma[k]*np.random.normal(loc=1) / comb(degrees[i],k) + (delta if len(S) == 1 and S[0] == i else 0)))
        C.append(np.array(Ci))

    return C

######## Potential Outcomes ########

def _z_tilde(Z,Ni,beta):
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
    return lambda Z: np.array([C[i].dot(_z_tilde(Z,G[i],beta)) for i in range(len(Z))]) if Z.ndim == 1 else \
        np.stack([C[i].dot(_z_tilde(Z,G[i],beta)) for i in range(Z.shape[0])], axis=1)

def outcomes(Z,G,C,beta,delta):

    # convert to adjacency matrix
    n = len(G)

    A = np.zeros((n,n))
    for i,Ni in enumerate(G):
        A[i,Ni] = 1

    #degrees_old = np.array([len(Ni) for Ni in G])
    degrees = np.sum(A,axis=1)
    #assert(np.all(degrees_old == degrees))
    

    #if Z.ndim == 1:
        #old_t = np.array([np.sum(Z[G[i]]) for i in range(n)]) # number of treated neighbors 
        #print(old_t)
    t = A @ Z # number of treated neighbors 
        #print(t)
        #assert(np.all(old_t == t))
        
    Y = np.zeros(n)
    for k in range(beta+1):
        Y +=  binom(t,k)/binom(degrees,k) * C[:,k]
    Y += delta * Z

    return Y
    
    # else: #Z is n x r
    #     print(Z.shape)
    #     Y = np.zeros_like(Z)

    #     t = Z @ A 
    #     for k in range(beta+1):
    #         Y += C[:,k] * binom(t,k)/binom(degrees,k)
    #     Y += delta * Z

    #     return Y


def pom_ugander_yin(G,h,beta):
    '''
    Returns vectors of coefficients c_i,S for each individual i and neighborhood subset S 
    Coefficients are given by a modification of Ugander/Yin's model to incorporate varied treatment effects across individuals and higher-order neighbor effects
        G = adjacency list representation of graph
        h = vector of (scaled) homophily effects
        beta = model degree
    '''

    # parameters 
    a = 1                                         # baseline effect
    sigma = 0.1                                   # magnitude of random perturbation on baselines
    delta = 1                                     # magnitude of direct effect
    gamma = [0.5**(k-1) for k in range(beta+1)]   # magnitude of subset treatment effects
    tau = 0.01                                    # magnitude of random perturbation on treatment effects

    n = len(G)
    degrees = [len(Ni) for Ni in G]
    dbar = sum(degrees)/n

    baseline = ( a + h + sigma * np.random.normal(size=n) ) * degrees/dbar

    C = np.empty((n,beta+1)) # C[i,k] = uniform effect coefficient c_i,S for |S| = k, excluding individual boost delta
    C[:,0] = baseline

    for k in range(1,beta+1):
        C[:,k] = baseline * (gamma[k] + tau * np.random.normal(size=n))

    return lambda Z : outcomes(Z,G,C,beta,delta)

######## Treatment Assignments ########

def staggered_rollout_two_stage(n,Cl,poq,Q,r=1):
    '''
        Returns treatment samples from Bernoulli staggered rollout
        n = number of individuals
        Cl = clusters, list of lists that partition [n]
        poq = cluster selection probability
        Q = treatment probabilities of selected units for each time step: beta+1
        r = number of replications
    '''
    k = len(Cl)

    T = np.zeros((n,k))
    for (j,cl) in enumerate(Cl):
        T[cl,j] = 1

    selection_mask = T @ ((np.random.rand(k,r) < poq) + 0)

    Z = np.zeros((len(Q),n,r))
    U = np.random.rand(n,r)     # random values that determine when individual i starts being treated

    for t in range(len(Q)):
        Z[t,:,:] = (U < Q[t]) + 0

    return (Z * selection_mask, selection_mask)

######## Estimator ########

def _interp_coefficients(P):
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

def pi_estimate_tte_two_stage(Z,Y,poq,Q):
    '''
    Returns TTE estimate from polynomial interpolation
        Z = treatment assignments: (beta+1) x n x r
        Y = potential outcomes function: {0,1}^n -> R^n
        poq = cluster selection probability
        Q = treatment probabilities of selected units for each time step: beta+1
    '''
    T,n,r = Z.shape

    H = _interp_coefficients(Q)

    TTE_hat = np.zeros(r)
    for t in range(T):
        TTE_hat += H[t]*np.sum(Y(Z[t,:]),axis=1)
    
    return TTE_hat/(n*poq)

    
