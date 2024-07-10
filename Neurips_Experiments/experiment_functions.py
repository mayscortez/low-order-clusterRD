import numpy as np
from numpy.random import RandomState
from scipy.special import binom
import scipy

rng = RandomState(19025)

######## Constructed Networks ########

def sbm(n,k,pii,pij):
    ''' 
    Returns a graph sampled from a stochastic block model
        n = number of vertices
        k = number of communities
        pii = edge probability within community
        pij = edge probability across communities
    '''

    c = n//k  # community size

    A = (rng.rand(n,n) < pij) + 0
    for i in range(k):
        A[i*c:(i+1)*c,i*c:(i+1)*c] = (rng.rand(c,c) < pii) + 0

    A[range(n),range(n)] = 1   # everyone is affected by their own treatment

    return scipy.sparse.csr_matrix(A)

######## Potential Outcomes Model ########

def homophily_effects(G):
    '''
    Returns vectors of (normalized) homophily effects as described in the Ugander/Yin paper
        G = adjacency list representation of graph
    '''
    n = G.shape[0]
    degrees = G.sum(axis=0)
    normalized_laplacian = (-G/degrees).tocsr()
    normalized_laplacian[range(n),range(n)] = 1

    _,eigvecs_s = scipy.sparse.linalg.eigs(normalized_laplacian,k=2,which='SM')
    h = eigvecs_s[:,1].real

    h = 2*(h-min(h))/(max(h)-min(h)) - 1   # adjust so smallest value is -1 and largest value is 1

    return h

def _outcomes(Z,G,C,d,beta,delta):
    '''
    Returns a matrix of outcomes for the given tensor of treatment assignments
        Z = treatment assignments: (beta+1) x r x n
        G = adjacency list representation of graph
        C = Ugander Yin coefficients: (beta+1) x n
        d = vector of vertex degrees: n 
        beta = model degree
        delta = magnitide of direct
    '''

    if Z.ndim == 3:
        t = np.empty_like(Z)
        for b in range(Z.shape[0]):
            t[b,:,:] = Z[b,:,:] @ G
    else:
        t = Z @ G           # number of treated neighbors 

    Y = delta * Z
    for k in range(beta+1):
        Y += (binom(t,k) / np.where(d>k,binom(d,k),1)) * C[k,:]

    return Y

def pom_ugander_yin(G,h,beta):
    '''
    Returns vectors of coefficients c_i,S for each individual i and neighborhood subset S 
    Coefficients are given by a modification of Ugander/Yin's model to incorporate varied treatment effects across individuals and higher-order neighbor effects
        G = adjacency list representation of graph
        d = vector of vertex degrees
        h = vector of homophily effects
        beta = model degree
    '''

    # parameters 
    a = 1                                         # baseline effect
    b = 0.5                                       # magnitude of homophily effects on baselines
    sigma = 0.1                                   # magnitude of random perturbation on baselines
    delta = 0.5                                   # magnitude of direct effect
    gamma = [0.5**(k-1) for k in range(beta+1)]   # magnitude of subset treatment effects
    tau = 0                                       # magnitude of random perturbation on treatment effects

    n = G.shape[0]
    d = np.ones(n) @ G         # vertex degrees
    dbar = np.sum(d)/n

    baseline = ( a + b * h + sigma * rng.normal(size=n) ) * d/dbar

    C = np.empty((beta+1,n)) # C[k,i] = uniform effect coefficient c_i,S for |S| = k, excluding individual boost delta
    C[0,:] = baseline

    for k in range(1,beta+1):
        C[k,:] = baseline * (gamma[k] + tau * rng.normal(size=n))

    return lambda Z : _outcomes(Z,G,C,d,beta,delta)

######## Treatment Assignments ########

def staggered_rollout_two_stage(n,Cl,p,Q,r=1):
    '''
        Returns treatment samples from Bernoulli staggered rollout: (beta+1) x r x n
        n = number of individuals
        Cl = clusters, list of lists that partition [n]
        p = treatment budget
        Q = treatment probabilities of selected units for each time step: beta+1
        r = number of replications
    '''
    if len(Cl) == 0:
        Z,U = staggered_rollout_two_stage_unit(n,p,Q,r)
        return (Z,U)
    else:
        k = len(Cl)

        T = np.zeros((k,n))
        for (j,cl) in enumerate(Cl):
            T[j,cl] = 1

        selection_mask = ((rng.rand(r,k) < p/Q[-1]) + 0) @ T

        Z = np.zeros((len(Q),r,n))
        U = rng.rand(r,n)     # random values that determine when individual i starts being treated

        for t in range(len(Q)):
            Z[t,:,:] = (U < Q[t]) + 0

        return (Z * selection_mask, selection_mask)

def staggered_rollout_two_stage_unit(n,p,Q,r=1):
    '''
        Returns treatment samples from Bernoulli staggered rollout: (beta+1) x r x n
        n = number of individuals
        p = treatment budget
        Q = treatment probabilities of selected units for each time step: beta+1
        r = number of replications
    '''

    Z = np.zeros((len(Q),r,n))
    V = rng.rand(r,n)     # random values that determine when individual i starts being treated
    U = (V < p/Q[-1]) + 0

    for t in range(len(Q)):
        Z[t,:,:] = (V < Q[t]*p/Q[-1]) + 0

    return (Z,U)

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

def pi_estimate_tte_two_stage(Y,p,Q):
    '''
    Returns TTE estimate from polynomial interpolation
        Y = potential outcomes: (beta+1) x r x n
        poq = cluster selection probability
        Q = treatment probabilities of selected units for each time step: beta+1
    '''
    n = Y.shape[-1]
    H = _interp_coefficients(Q)
    
    return 1/(n*p/Q[-1]) * H @ np.sum(Y,axis=-1)

def two_stage_restricted_estimator(Y,U,p,Q):
    '''
    Returns TTE estimate from polynomial interpolation
        Y = potential outcomes: (beta+1) x r x n
        U = selected individuals: r x n
        p = treatment budget
        Q = treatment probabilities of selected units for each time step: beta+1
    '''
    n = Y.shape[-1]
    H = _interp_coefficients(Q)
    
    return 1/(n*p/Q[-1]) * H @ np.sum(Y*U,axis=-1)


######## Other Estimators ########

def dm_estimate_tte(Z,Y):
    '''
    Returns TTE estimate from difference in means
        Z = treatment assignments: r x n
        Y = potential outcomes: r x n
    '''
    T,_,n = Z.shape

    num_treated = Z.sum(axis=2)

    DM_data = np.sum(Y*Z,axis=2)/np.maximum(num_treated,1)          # (beta+1) x r
    DM_data -= np.sum(Y*(1-Z),axis=2)/np.maximum(n-num_treated,1)  
    return np.sum(DM_data,axis=0)/T


def dm_threshold_estimate_tte(Z,Y,G,gamma):
    '''
    Returns TTE estimate from a thresholded difference in means
    i is "sufficiently treated" if they are treated, along with a (1-gamma) fraction of Ni
    i is "sufficiently control" if they are control, along with a (1-gamma) fraction of Ni
        Z = treatment assignments: (beta+1) x r x n
        Y = potential outcomes function: {0,1}^n -> R^n
        G = causal network (this estimator is not graph agnostic)
        gamma = tolerance parameter (as described above)
    '''

    T,_,n = Z.shape

    d = np.ones(n) @ G                             # vertex degrees
    num_Ni_treated = np.empty_like(Z)              # number of treated neighbors
    for t in range(T):
        num_Ni_treated[t,:,:] = Z[t,:,:] @ G
    frac_Ni_treated = num_Ni_treated / d

    sufficiently_treated = (frac_Ni_treated >= (1-gamma)) * Z
    num_sufficiently_treated = sufficiently_treated.sum(axis=2)
    sufficiently_control = (frac_Ni_treated <= gamma) * (1-Z)
    num_sufficiently_control = sufficiently_control.sum(axis=2)

    DM_data = np.sum(Y*sufficiently_treated,axis=2)/np.maximum(num_sufficiently_treated,1)          # (beta+1) x r
    DM_data -= np.sum(Y*sufficiently_control,axis=2)/np.maximum(num_sufficiently_control,1)  
    return np.sum(DM_data,axis=0)/T

def _neighborhood_cluster_sizes(N,Cl):
    '''
    Returns a list which, for each i, has an array of its number of neighbors from each cluster
        N = neighborhoods, list of adjacency lists
        Cl = clusters, list of lists that partition [n]
    '''
    n = len(N)

    membership = np.zeros(n,dtype=np.uint32)
    for i,C in enumerate(Cl):
        membership[C] = i

    neighborhood_cluster_sizes = np.zeros((n,len(Cl)))
    for i in range(n):
        for j in N[i]:
            neighborhood_cluster_sizes[i,membership[j]] += 1
    
    return neighborhood_cluster_sizes

def ht_hajek_estimate_tte(Z,Y,G,Cl,p,q):
    '''
    Returns TTE Horvitz-Thompson/Hajek estimates
        Z = treatment assignments: r x n
        Y = potential outcomes function: {0,1}^n -> R^n
        G = causal network (this estimator is not graph agnostic)
        Cl = clusters, list of lists that partition [n]
        p = treatment budget
        q = treatment probabilities in selected clusters
    '''
    _,n = Z.shape

    N = []
    for i in range(n):
        N.append(G[:,[i]].nonzero()[0])

    ncs = _neighborhood_cluster_sizes(N,Cl)
    d = ncs.sum(axis=1)               # degree
    cd = np.count_nonzero(ncs,axis=1) # cluster degree

    Ni_fully_treated = np.empty_like(Z)
    for i in range(n):
        Ni_fully_treated[:,i] = np.prod(Z[:,N[i]],axis=1)

    Ni_fully_control = np.empty_like(Z)
    for i in range(n):
        Ni_fully_control[:,i] = np.prod(1-Z[:,N[i]],axis=1)

    prob_fully_treated = np.power(p/q,cd) * np.power(q,d)
    prob_fully_control = np.prod(1 - p/q*(1-np.power(1-q,ncs)),axis=1)

    HT_data = (np.sum(Y * Ni_fully_treated/prob_fully_treated, axis=1) - np.sum(Y * Ni_fully_control/prob_fully_control, axis=1))/n

    nhat1 = np.sum(Ni_fully_treated/prob_fully_treated, axis=1)
    nhat2 = np.sum(Ni_fully_control/prob_fully_control, axis=1)

    Hajek_data = np.sum(Y * Ni_fully_treated/prob_fully_treated, axis=1)/nhat1
    Hajek_data -= np.sum(Y * Ni_fully_control/prob_fully_control, axis=1)/nhat2  
    return (HT_data,Hajek_data)

######## Utility Function for Computing Effect Sizes ########

def e(n,S):
    v = np.zeros(n)
    v[S] = 1
    return v

def LPis(fY,Cl,n):
    L = {}

    for i,C1 in enumerate(Cl):
        L[frozenset([i])] = np.sum(fY(e(n,C1)) - fY(np.zeros(n)))

        for ip,C2 in enumerate(Cl):
            if ip >= i: continue
            L[frozenset([i,ip])] = np.sum(fY(e(n,list(C1)+list(C2))) - fY(e(n,C1)) - fY(e(n,C2)) + fY(np.zeros(n)))

    return L