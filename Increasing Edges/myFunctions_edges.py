import numpy as np
import networkx as nx
from scipy import interpolate
import scipy.sparse

# Scale down the effects of higher order terms
a1 = 1      # for linear effects
a2 = 1    # for quadratic effects
a3 = 1   # for cubic effects
a4 = 1   # for quartic effects

# Define f(z)
f_linear = lambda alpha, z, gz: alpha + a1*z
f_quadratic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz)
f_cubic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3)
f_quartic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3) + a4*np.power(gz,4)

def ppom(beta, C, alpha):
    '''
    Returns k-degree polynomial potential outcomes function fy
    
    f (function): must be of the form f(z) = alpha + z + a2*z^2 + a3*z^3 + ... + ak*z^k
    C (np.array): weighted adjacency matrix
    alpha (np.array): vector of null effects
    '''
    # n = C.shape[0]
    # assert np.all(f(alpha, np.zeros(n), np.zeros(n)) == alpha), 'f(0) should equal alpha'
    # assert np.all(np.around(f(alpha, np.ones(n)) - alpha - np.ones(n), 10) >= 0), 'f must include linear component'

    if beta == 0:
        return lambda z: alpha + a1*z
    elif beta == 1:
        f = f_linear
        return lambda z: alpha + a1*C.dot(z)
    else:
        g = lambda z : C.dot(z) / np.array(np.sum(C,1)).flatten()
        if beta == 2:
            f = f_quadratic
        elif beta == 3:
            f = f_cubic
        elif beta == 4:
            f = f_quartic
        else:
            print("ERROR: invalid degree")
        return lambda z: f(alpha, C.dot(z), g(z)) 

bernoulli = lambda n,p : (np.random.rand(n) < p) + 0

def SBM(n, k, Pii, Pij):
    '''
    Returns the adjacency matrix (as a scipy sparse array) of a stochastic block model on n nodes with k communities
    The edge prob within the same community is Pii
    The edge prob across different communities is Pij
    '''
    sizes = np.zeros(k, dtype=int) + n//k
    probs = np.zeros((k,k)) + Pij
    np.fill_diagonal(probs, Pii)
    G = nx.stochastic_block_model(sizes, probs, directed=True, selfloops=True)
    A = nx.to_scipy_sparse_array(G, format='coo')
    A.setdiag(1)
    return G, scipy.sparse.csr_array(A)

def binary_covariate_weights(nc, A, phi, mu1 = 1/2, mu2 = 2.5):
    ''' Returns weighted adjacency matrix, where weights depend on a binary covariate type

    C[i,j] ~ Normal(mu1*mu1, 0.5) if both i and j are type 1
    C[i,j] ~ Normal(mu2*mu2, 0.5) if both i and j are type 2
    C[i,j] ~ Normal(mu1*mu2, 0.5) if i and j are different types

    Parameters
    ----------
    nc : int
        number of clusters
    mu1 : float

    mu2: float
        
    phi : float
        probability of switching type
    A : scipy sparse csr array
        adjacency matrix

    Returns
    ---------
    C : scipy sparse csr array
        weighted adjacency matrix
    '''
    n = A.shape[0]
    means = np.ones(n)
    midpoint = int((nc//2)*(n/nc))
    means[0:midpoint] = mu1
    means[midpoint:] = mu2

    rng  = np.random.default_rng()
    switchers = rng.random(n)
    switchers = (switchers < phi) + 0 
    means = np.concatenate((np.where(switchers[0:midpoint]==1, mu2, means[0:midpoint]), np.where(switchers[midpoint:]==1, mu1, means[midpoint:])))
    means = np.outer(means, means)
    weights = rng.normal(loc=means, scale=0.5)

    return scipy.sparse.csr_array(A.multiply(weights))

def simpleWeights(A, diag=5, offdiag=5, rand_diag=np.array([]), rand_offdiag=np.array([])):
    '''
    Returns weights generated from simpler model

    A (numpy array): adjacency matrix of the network
    diag (float): maximum norm of direct effects
    offidiag (float): maximum norm of the indirect effects
    '''
    n = A.shape[0]

    if rand_offdiag.size == 0:
        rand_offdiag = np.random.rand(n)
    C_offdiag = offdiag*rand_offdiag

    in_deg = scipy.sparse.diags(np.array(A.sum(axis=1)).flatten(),0)  # array of the in-degree of each node
    C = in_deg.dot(A - scipy.sparse.eye(n))
    col_sum = np.array(C.sum(axis=0)).flatten()
    col_sum[col_sum==0] = 1
    temp = scipy.sparse.diags(C_offdiag/col_sum)
    C = C.dot(temp)

    if rand_diag.size == 0:
        rand_diag = np.random.rand(n)
    C_diag = diag*rand_diag
    C.setdiag(C_diag)

    return C

def select_clusters_bernoulli(numOfClusters, q):
    '''
    Assumes clusters are labeled 0,1,2,...,numOfClusters-1 and randomly chooses clusters according to a Bernoulli(q) design
    
    Parameters
    ------------
    numOfClusters : int
        NC=nc**2; the total number of clusters; given population size n*n, NC = ceil(n/k)**2 where k is the cluster side length
    q : float
        fraction of clusters you wish to select (in expectation)

    Returns
    --------
    selected : numpy array
        array of the labels of the randomly selected clusters

    '''

    design = (np.random.rand(numOfClusters) < q) + 0
    selected = np.where(design == 1)[0]
    return selected

def select_clusters_complete(numOfClusters, K):
    '''
    Assumes clusters are labeled 0,1,2,...,numOfClusters-1 and samples K clusters uniformly at random from all subsets of clusters of size K 
    (i.e. according to completely randomized design)
    
    Parameters
    ------------
    numOfClusters : int
        NC=nc**2; the total number of clusters; given population size n*n, NC = ceil(n/k)**2 where k is the cluster side length
    K : int
        number of clusters you wish to select

    Returns
    --------
    selected : numpy array
        array of the labels of the randomly selected clusters

    '''
    design = np.zeros(shape=(numOfClusters))
    design[0:K] = np.ones(shape=(K))
    rng = np.random.default_rng()
    rng.shuffle(design)
    selected = np.where(design==1)[0]
    
    return selected

def zU_to_z(z_U, U, z_U_prime, Uprime, n):
    '''
    Let U be the set of individuals whose clusters were chosen to be randomized to treatment or control.
    Let Uprime be the set of in-neighbors of nodes in U who are not themselves in U (i.e. the boundary of U)
    This function takes the treatment assignment vector of U and Uprime and returns
    the treatment assignment vector for the whole population of size n.

    Parameters
    -----------
    z_U : array
        treatment assignment vector for nodes in U
    U : list
        list of the nodes in U
    z_U_prime : array
        treatment assignment vector for nodes in Uprime
    U : list
        list of the nodes in Uprime   
    n : int
        size of the popluation
    '''
    # Get the indices from [n_U] and [n_{Uprime}] of treated units
    treated_U = np.nonzero(z_U)[0]
    treated_U_prime = np.nonzero(z_U_prime)[0]

    # Get their corresponded indices in [N]
    treated_U = list(map(U.__getitem__, treated_U))
    treated_U_prime = list(map(Uprime.__getitem__, treated_U_prime))
    treated = treated_U + treated_U_prime
    
    # Create the treatment assignment vector of the whole population
    z = np.zeros(n)
    np.put(z,treated,1)

    return z

def staggered_rollout_bern(n, P):
  '''
  Returns Treatment Samples from Bernoulli Staggered Rollout

  beta (int): degree of potential outcomes model
  n (int): size of population
  P (numpy array): treatment probabilities for each time step
  '''

  ### Initialize ###
  Z = np.zeros(shape=(P.size,n))   # for each treatment sample z_t
  U = np.random.rand(n)

  ### staggered rollout experiment ###
  for t in range(P.size):
    ## sample treatment vector ##
    Z[t,:] = (U < P[t])+0

  return Z

def staggered_rollout_bern_clusters(n, selected, P, bndry, P_prime):
  '''
  Returns Treatment Samples from Bernoulli Staggered Rollout with clustering

  n (int): size of population
  selected (list): list of the nodes who were selected to be in the staggered rollout experiment
  P (numpy array): treatment probabilities for each time step for the selected group
  bndry (list): boundary of selected (neighbors of nodes in selected who are not themselves selected)
  P_prime (numpy array): treatment probabilities for each time step for the boundary group
  '''

  ### Initialize ###
  T = len(P)
  Z = np.zeros(shape=(T,n))   # for each treatment sample z_t
  W = np.random.rand(len(selected))
  W_prime = np.random.rand(len(bndry))

  ### staggered rollout experiment ###
  for t in range(T):
    ## sample treatment vector ##
    z_U = (W < P[t])+0
    z_U_prime = (W_prime < P_prime[t])+0
    Z[t,:] = zU_to_z(z_U, selected, z_U_prime, bndry, n)

  return Z

def bern_coeffs(P):
  '''
  Returns Coefficients h_t from Bernoulli Staggered Rollout

  P (numpy array): treatment probabilities for each time step
  '''

  ### Initialize ###
  H = np.zeros(P.size)

  ### Coefficients ###
  for t in range(P.size):
    one_minusP = 1 - P            # [1-p0, 1-p1, ... , 1-p_beta]
    pt_minusP = P[t] - P          # [pt-p0, pt-p1, ... , pt-p_beta]
    minusP = -1*P                 # [-p0, -p1, ... , -p_beta]
    one_minusP[t] = 1; pt_minusP[t] = 1; minusP[t] = 1
    fraction1 = one_minusP/pt_minusP
    fraction2 = minusP/pt_minusP
    H[t] = np.prod(fraction1) - np.prod(fraction2)

  return H

def seq_treatment_probs(beta, p):
  '''
  Returns sequence of treatment probabilities for Bernoulli staggered rollout

  beta (int): degree of the polynomial; order of interactions of the potential outcomes model
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05
  '''
  fun = lambda i: (i)*p/(beta)
  P = np.fromfunction(fun, shape=(beta+1,))
  return P

def outcome_sums(n, Y, Z, selected):
    '''
    Returns the sums of the outcomes Y(z_t) for each timestep t

    Y (function): potential outcomes model
    Z (numpy array): treatment vectors z_t for each timestep t
    - each row should correspond to a timestep, i.e. Z should be beta+1 by n
    selected (list): indices of units in the population selected to be part of the experiment (i.e in U)
    '''
    if len(selected) == n: # if we selected all nodes, sums = sums_U
        sums = np.zeros(Z.shape[0])
        for t in range(Z.shape[0]):
            outcomes = Y(Z[t,:])
            sums[t] = np.sum(outcomes)
        return sums, sums
    else: 
        sums, sums_U = np.zeros(Z.shape[0]), np.zeros(Z.shape[0]) 
        for t in range(Z.shape[0]):
            outcomes = Y(Z[t,:])
            sums[t] = np.sum(outcomes)
            sums_U[t] = np.sum(outcomes[selected])
    return sums, sums_U

def PI(n, sums, H):
    '''
    Returns an estimate of the TTE with (beta+1) staggered rollout design

    n (int): popluation size
    H (numpy array): PPOM coefficients h_t or l_t
    sums (numpy array): sums of outcomes at each time step
    '''
    if n > 0:
        return (1/n)*H.dot(sums)
    else:
        return 0

def poly_interp_splines(n, P, sums, spltyp = 'quadratic'):
  '''
  Returns estimate of TTE using spline polynomial interpolation 
  via scipy.interpolate.interp1d

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  spltyp (str): type of spline, can be 'quadratic, or 'cubic'
  '''
  assert spltyp in ['quadratic', 'cubic'], "spltyp must be 'quadratic', or 'cubic'"
  f_spl = interpolate.interp1d(P, sums, kind=spltyp, fill_value='extrapolate')
  TTE_hat = (1/n)*(f_spl(1) - f_spl(0))
  return TTE_hat

def poly_interp_linear(n, P, sums):
  '''
  Returns two estimates of TTE using linear polynomial interpolation 
  via scipy.interpolate.interp1d
  - the first is with kind = 'linear' (as in... ?)
  - the second is with kind = 'slinear' (as in linear spline)

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  '''

  #f_lin = interpolate.interp1d(P, sums, fill_value='extrapolate')
  f_spl = interpolate.interp1d(P, sums, kind='slinear', fill_value='extrapolate')
  #TTE_hat1 = (1/n)*(f_lin(1) - f_lin(0))
  TTE_hat2 = (1/n)*(f_spl(1) - f_spl(0))
  #return TTE_hat1, TTE_hat2
  return TTE_hat2


def poly_LS_prop(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)/(np.array(A.sum(axis=1)).flatten()-1+1e-10)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v)-v[0]

def poly_LS_num(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  # least squares regression
  v = np.linalg.lstsq(X,y,rcond=None)[0]

  # Estimate TTE
  count = 1
  treated_neighb = np.array(A.sum(axis=1)).flatten()-1
  for i in range(beta):
      X[:,count] = np.power(treated_neighb,i)
      X[:,count+1] = np.power(treated_neighb,i+1)
      count += 2
  TTE_hat = np.sum((X @ v) - v[0])/n
  return TTE_hat

def DM_naive(y, z):
    '''
    Returns an estimate of the TTE using difference in means
    (mean outcome of individuals in treatment) - (mean outcome of individuals in control)

    y (numpy array): observed outcomes
    z (numpy array): treatment vector
    '''
    treated = np.sum(z)
    untreated = np.sum(1-z)
    est = 0
    if treated > 0:
        est = est + y.dot(z)/treated
    if untreated > 0:
        est = est - y.dot(1-z)/untreated
        return est

def DM_fraction(n, y, A, z, tol):
    '''
    Returns an estimate of the TTE using weighted difference in means where 
    we only count neighborhoods with at least tol fraction of the neighborhood being
    assigned to treatment or control

    n (int): number of individuals
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    tol (float): neighborhood fraction treatment/control "threshhold"
    '''
    z = np.reshape(z,(n,1))
    treated = 1*(A.dot(z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    treated = np.multiply(treated,z).flatten()
    control = 1*(A.dot(1-z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    control = np.multiply(control,1-z).flatten()

    est = 0
    if np.sum(treated) > 0:
        est = est + y.dot(treated)/np.sum(treated)
    if np.sum(control) > 0:
        est = est - y.dot(control)/np.sum(control)
    return est

def horvitz_thompson(n, nc, y, A, z, q, p):
    '''Computes the Horvitz-Thompson estimate of the TTE under Bernoulli design or Cluster-Bernoulli design.
    
    Parameters
    ----------
    n : int
        the size of the population/network
    nc : int
        the number of clusters (equals n if simple Bernoulli design with no clustering)
    y : numpy array
        the outcomes of each unit in the population
    A : scipy sparse array
        adjacency matrix of the network such that A[i,j]=1 indicates that unit j is an in-neighbor of i
    z : numpy array
        the treatment assignment of each unit in the population
    q : float
        probability that a cluster is indepdently chosen for treatment (should equal 1 under simple Bernoulli design with no clustering)
    p : float
        the treatment probability for chosen clusters in the staggered rollout
    '''
    AA = A.toarray()

    cluster = []
    for i in range(1,nc+1):
        cluster.extend([i]*(n//nc))

    cluster_neighborhoods = np.apply_along_axis(lambda x: np.bincount(x*cluster, minlength=nc+1), axis=1, arr=AA)[:,1:]
    
    degree = np.sum(cluster_neighborhoods, axis=1)
    cluster_degree = np.count_nonzero(cluster_neighborhoods, axis=1)

    # Probabilities of each person's neighborhood being entirely treated or entirely untreated
    all_treated_prob = np.power(p, degree) * np.power(q, cluster_degree)
    none_treated_prob = np.prod(np.where(cluster_neighborhoods>0,(1-q)+np.power(1-p,cluster_neighborhoods)*q,1),axis=1)

    # Indicators of each person's neighborhood being entirely treated or entirely untreated
    all_treated = np.prod(np.where(AA>0,z,1),axis=1)
    none_treated = np.prod(np.where(AA>0,1-z,1),axis=1)

    zz = np.nan_to_num(np.divide(all_treated,all_treated_prob) - np.divide(none_treated,none_treated_prob))

    return 1/n * y.dot(zz)