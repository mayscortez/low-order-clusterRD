import numpy as np
from itertools import combinations
from math import comb
from scipy.special import binom

######## Potential Outcomes Model ########

def homophily_effects(G):
    '''
    Returns vectors of (normalized) homophily effects as described in the Ugander/Yin paper
        G = adjacency list representation of graph
    '''
    n = G.shape[0]
    degrees = np.sum(G,axis=1)
    normalized_laplacian = -G/degrees
    normalized_laplacian[range(n),range(n)] = 1
    
    eigvals, eigvecs = np.linalg.eig(normalized_laplacian)
    fiedler_index = np.where(eigvals.real == np.sort(eigvals.real)[1])[0][0]
    h = eigvecs[:,fiedler_index].real

    return h/(max(abs(max(h)),abs(min(h))))

def _outcomes(Z,G,C,beta,delta):
    '''
    Returns a matrix of outcomes for the given tensor of treatment assignments
        Z = treatment assignments: (beta+1) x r x n
        G = adjacency list representation of graph
        C = Ugander Yin coefficients: (beta+1) x n
        beta = model degree
        delta = magnitide of direct
    '''
    n = G.shape[0]
    d = np.sum(G,axis=1)   # degrees
    t = Z @ G.T            # number of treated neighbors 
        
    Y = delta * Z
    for k in range(beta+1):
        Y += (binom(t,k) / np.where(d>k,binom(d,k),1)) * C[k,:]

    return Y

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

    n = G.shape[0]
    degrees = np.sum(G,axis=1)
    dbar = sum(degrees)/n

    baseline = ( a + h + sigma * np.random.normal(size=n) ) * degrees/dbar

    C = np.empty((beta+1,n)) # C[k,i] = uniform effect coefficient c_i,S for |S| = k, excluding individual boost delta
    C[0,:] = baseline

    for k in range(1,beta+1):
        C[k,:] = baseline * (gamma[k] + tau * np.random.normal(size=n))

    return lambda Z : _outcomes(Z,G,C,beta,delta)


######## Treatment Assignments ########

def staggered_rollout_two_stage(n,Cl,poq,Q,r=1):
    '''
        Returns treatment samples from Bernoulli staggered rollout: (beta+1) x r x n
        n = number of individuals
        Cl = clusters, list of lists that partition [n]
        poq = cluster selection probability
        Q = treatment probabilities of selected units for each time step: beta+1
        r = number of replications
    '''
    k = len(Cl)

    T = np.zeros((k,n))
    for (j,cl) in enumerate(Cl):
        T[j,cl] = 1

    selection_mask = ((np.random.rand(r,k) < poq) + 0) @ T

    Z = np.zeros((len(Q),r,n))
    U = np.random.rand(r,n)     # random values that determine when individual i starts being treated

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
        Z = treatment assignments: (beta+1) x r x n
        Y = potential outcomes function: {0,1}^n -> R^n
        poq = cluster selection probability
        Q = treatment probabilities of selected units for each time step: beta+1
    '''
    n = Z.shape[-1]
    H = _interp_coefficients(Q)
    
    return 1/(n*poq) * H @ np.sum(Y(Z),axis=-1)

    
