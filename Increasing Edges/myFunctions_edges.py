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

def weights_im_normal(n, d=1, sigma=0.1, neg=0):
    '''
    Returns weights C (numpy array) under the influence and malleability framework,
    with Gaussian mean-zero noise.

    n (int): number of individuals
    d (int): number of influence dimensions
    sigma (float): standard deviation of noise
    neg (0 or 1): 0 if restricted to non-negative weights, 1 otherwise
    '''
    X = np.random.rand(d,n)              # influence
    W = np.random.rand(d,n)              # malliability

    if neg==0:
      E = np.abs(np.random.normal(scale=sigma, size=(n,n)))
    else:
      E = np.random.normal(scale=sigma, size=(n,n))
    C = X.T.dot(W)+E
    return C

def normalized_weights(C, diag=10, offdiag=8):
    '''
    Returns normalized weights (or normalized weighted adjacency matrix) as numpy array

    C (square numpy array): weight matrix (or weighted adjacency matrix)
    diag (float): controls the magnitude of the diagonal elements
    offdiag (float): controls the magnitude of the off-diagonal elements
    '''
    n = C.shape[0]

    # diagonal elements
    C_diag = np.ones(n) * diag * np.random.rand(n)

    # remove diagonal elements and normalize off-diagonal elements
    # normalizes each column by the norm of the column (not including the diagonal element)
    np.fill_diagonal(C, 0)
    col_norms = np.linalg.norm(C, axis=0, ord=1)
    col_norms = np.where(col_norms != 0, col_norms, col_norms+1)
    C = (C / col_norms) * offdiag * np.random.rand(n)

    # add back the diagonal
    C += np.diag(C_diag)

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
    selected = np.where(design == 1)
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

def staggered_rollout_bern(n, selected, P, bndry, P_prime):
  '''
  Returns Treatment Vectors from Bernoulli Staggered Rollout for each time step 

  n (int): size of population
  selected (list): list of the nodes who were selected to be in the staggered rollout experiment
  P (numpy array): treatment probabilities for each time step for the selected group
  bndry (list): boundary of selected (neighbors of nodes in selected who are not themselves selected)
  P_prime (numpy array): treatment probabilities for each time step for the boundary group

  Z (numpy array): treatment vector for all n population units for each time step in the staggered rollout
  ZU (numpy array): treatment vector for only selected population units (U) for each time step in the staggered rollout
  '''

  ### Initialize ###
  T = len(P)        # number of time steps
  U = len(selected) # the size of U aka the number of selected nodes
  Z = np.zeros(shape=(T,n))     # for each treatment sample z_t in {0,1}^n
  #ZU = np.zeros(shape=(T,U))    # for each treatment sample z_t in {0,1}^|U|
  W = np.random.rand(len(selected))
  W_prime = np.random.rand(len(bndry))

  ### staggered rollout experiment ###
  for t in range(T):
    ## sample treatment vector ##
    z_U = (W < P[t])+0
    z_U_prime = (W_prime < P_prime[t])+0
    Z[t,:] = zU_to_z(z_U, selected, z_U_prime, bndry, n)
    #ZU[t,:] = z_U

  return Z#,ZU

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

def outcome_sums(Y, Z, selected):
  '''
  Returns the sums of the outcomes Y(z_t) for each timestep t

  Y (function): potential outcomes model
  Z (numpy array): treatment vectors z_t for each timestep t
   - each row should correspond to a timestep, i.e. Z should be beta+1 by n
  selected (list): indices of units in the population selected to be part of the experiment (i.e in U)
  '''
  sums, sums_U = np.zeros(Z.shape[0]), np.zeros(Z.shape[0])  
  for t in range(Z.shape[0]):
    outcomes = Y(Z[t,:])
    sums[t] = np.sum(outcomes)
    sums_U[t] = np.sum(outcomes[selected])
  return sums, sums_U

def graph_agnostic(n, sums, H):
    '''
    Returns an estimate of the TTE with (beta+1) staggered rollout design

    n (int): popluation size
    H (numpy array): PPOM coefficients h_t or l_t
    sums (numpy array): sums of outcomes at each time step
    '''
    return (1/n)*H.dot(sums)

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


def poly_regression_prop(beta, y, A, z):
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

def poly_regression_num(beta, y, A, z):
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

def diff_in_means_naive(y, z):
    '''
    Returns an estimate of the TTE using difference in means
    (mean outcome of individuals in treatment) - (mean outcome of individuals in control)

    y (numpy array): observed outcomes
    z (numpy array): treatment vector
    '''
    return y.dot(z)/np.sum(z) - y.dot(1-z)/np.sum(1-z)

def diff_in_means_fraction(n, y, A, z, tol):
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