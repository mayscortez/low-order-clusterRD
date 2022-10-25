import numpy as np
import networkx as nx
import scipy.sparse
from sklearn.cluster import SpectralClustering

'''
Potential Outcomes Models
- linear_pom: 
- linear_add1: Linear Additive Model I from Gui, Xu, Bhasin, Han (2015) paper
- linear_add2: Linear Additive Model II from Gui, Xu, Bhasin, Han (2015) paper
- ppom:
'''

# Linear Models
linear_pom = lambda C,alph,z : C.dot(z) + alph

linear_add1 = lambda alph,beta,gam,A,z : alph + beta*z + gam*((A.dot(z) - z) / (np.array(A.sum(axis=1)).flatten()-1+1e-10))

def linear_add2(alp0, alp1, gam0, gam1, A, z):
    '''
    alp0,alp1 (float): baseline effect if unit i is not treated / treated
    gam0,gam1 (float): network effect if unit is is not treated / treated
    A (scipy sparse matrix): adjacency matrix of the network
    z (numpy array): treatment vector
    '''
    frac_treated = (A.dot(z) - z) / (np.array(A.sum(axis=1)).flatten()-1+1e-10)
    indicator0 = (z==0)*1
    indicator1 = (z==1)*1
    lam2 = (indicator0 * (alp0 + gam0*frac_treated)) + (indicator1 * (alp1 + gam1*frac_treated))
    return lam2

# Polynomial (in z) models
def ppom(beta, C, alpha):
  '''
  Returns beta-degree polynomial potential outcomes function fy for beta in {0,1,2,3,4}
  
  f (function): must be of the form f(z) = alpha + z + a2*z^2 + a3*z^3 + ... + ak*z^beta
  C (np.array): weighted adjacency matrix
  alpha (np.array): vector of null/baseline effects
  '''
  f_quadratic = lambda alpha, z, gz: alpha + z + np.multiply(gz,gz)
  f_cubic = lambda alpha, z, gz: alpha + z + np.multiply(gz,gz) + np.power(gz,3)
  f_quartic = lambda alpha, z, gz: alpha + z + np.multiply(gz,gz) + np.power(gz,3) + np.power(gz,4)

  if beta == 0:
      return lambda z: alpha + z
  elif beta == 1:
      return lambda z: alpha + C.dot(z)
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

'''
Networks + Edge Weights
- erdos_renyi
- config_model_nx
- small_world
- lattice2Dsq
- rand_points_2D
- uniform_degrees
- powerlaw_degrees
- simpleWeights
'''

def erdos_renyi(n,p):
    '''
    Generates a random network of n nodes using the Erdos-Renyi method,
    where the probability that an edge exists between two nodes is p.

    Returns the adjacency matrix of the network as a scipy.sparse array
    '''
    A = np.random.rand(n,n)
    A = (A < p) + 0
    A[range(n),range(n)] = 1   # everyone is affected by their own treatment
    return scipy.sparse.csr_array(A)

def config_model_nx(N, exp = 2.5, law = "out"):
    '''
    Returns the adjacency matrix A (as a numpy array) of a networkx configuration
    model with power law degree and uniform degree sequences

    N (int): number of nodes
    law (str): inicates whether in-, out- or both in- and out-degrees should be distributed as a power law
        "out" : out-degrees distributed as powerlaw, in-degrees sum up to same # as out-degrees
        "in" : in-degrees distributed as powerlaw, out-degrees sum up to same # as in-degrees
        "both" : both in- and out-degrees distributed as powerlaw
    '''
    assert law in ["out", "in", "both"], "law must = 'out', 'in', or 'both'"
    if law == "out":
        deg_seq_out = powerlaw_degrees(N, exp)
        deg_seq_in = uniform_degrees(N,np.sum(deg_seq_out))

    elif law == "in":
        deg_seq_in = powerlaw_degrees(N, exp=2.5)
        deg_seq_out = uniform_degrees(N,np.sum(deg_seq_in))
    
    else:
        deg_seq_out = powerlaw_degrees(N, exp=2.5)
        deg_seq_in = powerlaw_degrees(N, exp=2.5)

        # This next part forces the two degree sequences to sum up to the same number
        # there is probably a more efficient way to do this
        sum = np.sum(deg_seq_out)
        i = 0
        while np.sum(deg_seq_in) < sum:
            deg_seq_in[i] += 1
            i += 1
        while np.sum(deg_seq_in) > sum:
            deg_seq_in[i] -= 1
            i += 1
        

    G = nx.generators.degree_seq.directed_configuration_model(deg_seq_in,deg_seq_out)

    G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops
    G = nx.DiGraph(G)                         # remove parallel edges
    A = nx.to_scipy_sparse_matrix(G)          # retrieve adjacency matrix
    A.setdiag(np.ones(N))                     # everyone is affected by their own treatment

    return A

def small_world(n,k,p):
    '''
    Returns adjacency matrix (A, numpy array) of random network using the Watts-
    Strogatz graph function in the networkx package.

    n (int): number of nodes
    k (int): Each node is joined with its k nearest neighbors in a ring topology
    p (float, in [0,1]): probability of rewiring each edge
    '''
    G = nx.watts_strogatz_graph(n, k, p)
    A = nx.to_scipy_sparse_matrix(G)                  # retrieve adjacency matrix
    A.setdiag(np.ones(n))                    # everyone is affected by their own treatment

    return A

def lattice2Dsq(x,y):
    '''
    Returns adjacency matrix of an x by y lattice graph on x*y nodes as a sparse matrix
    
    x (int): number of nodes in the x direction
    y (int): number of nodes in the y direction
    '''
    G = nx.grid_graph(dim=(x,y))
    G = nx.DiGraph(G)
    A = nx.to_scipy_sparse_matrix(G)
    A.setdiag(np.ones(x*y))
    return A 

def rand_points_2D(R_n, n):
    return R_n * np.random.uniform(-1,1,[n,2])

def powerlaw_degrees(N, exp):
    S_out = np.around(nx.utils.powerlaw_sequence(N, exponent=exp), decimals=0).astype(int)
    out_sum = np.sum(S_out)
    if (out_sum % 2 != 0):
        ind = np.random.randint(N)
        S_out[ind] += 1
    return S_out

def uniform_degrees(n,sum):
    '''
    Given n and sum, returns array whose entries add up to sum where each entry is in {sum/n, (sum,n)+1}
    i.e. to create uniform degrees for a network that add up to a specific number

    n: size of network
    sum: number that the entries of the array must add up to
    '''
    degs = (np.ones(n)*np.floor(sum/n)).astype(int)
    i = 0
    while np.sum(degs) != sum:
        degs[i] += 1
        i += 1
    return degs

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

'''
Cluster Randomized Experimental Designs
- lat_toCluster
- lat_toUnit
- spectralCRD: from Michael Leung "cluster_rand"
- bernoulli_cluster
'''
def lat_toCluster(I,J,k,q1=0,q2=0,divides = False):
  '''
  Returns the cluster assignment (s,t) of unit(s) (i,j) for i in I and j in J

  i (int or np.array): row position of unit on n by n lattice (or array of row positions)
  j (int or np.array): column position of unit on n by n lattice (or array of col positions)
  k (int): typical cluster side length (each cluster is itself a k by k grid graph with k << n)
  q1 (int): "origin" row position marking the END (inclusive) of first cluster
  q2 (int): "origin" col position marking the END (inclusive) of first cluster
  divides (boolean): if k divides n should be set to True
  '''
  if divides:
    s = np.floor(I/k)
    t = np.floor(J/k)
  else:
    s = np.ceil((I-q1)/k)
    t = np.ceil((J-q2)/k)
  
  return s.astype(int),t.astype(int)

def lat_toUnit(s,t,k,n,q1=0,q2=0):
  '''
  Returns the row indicdes I & column indices J (from the n by n lattice) corresponding to cluster (s,t)

  s (int): row position of cluster
  t (int): column position of cluster
  k (int): typical cluster side length (each cluster is itself a k by k grid graph with k << n)
  n (int): population is represented by an n by n square lattice/grid graph
  q1 (int): "origin" row position marking the end (inclusive) of first cluster
  q2 (int): "origin" col position marking the end (inclusive) of first cluster
  '''
  if n%k==0:
    starti = np.maximum(0,s*k)
    stopi = (s+1)*k - 1
    startj = np.maximum(0,t*k)
    stopj = (t+1)*k - 1
  else:
    starti = np.maximum(0,q1 + 1 + (s-1)*k)
    stopi = np.minimum(q1 + s*k, n-1)
    startj = np.maximum(0,q2 + 1 + (t-1)*k)
    stopj = np.minimum(q2 + t*k, n-1)
  
  I = np.linspace(starti,stopi,stopi-starti+1,dtype=int)
  J = np.linspace(startj,stopj,stopj-startj+1,dtype=int)
  
  return I,J

def spectralCRD(positions, num_clusters, p, seed):
    """ Generates spatial clusters and treatments via cluster randomization.

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n units.
    num_clusters : int
        number of clusters.
    p : float
        probability of assignment to treatment.
    seed : int
        set seed for k-means clustering initialization.

    Returns
    -------
    D : numpy array
        n x 1 array of treatment indicators, one for each of the n units.
    clusters : numpy array
        n x 1 array of cluster assignments, one for each of the n units. Clusters are labeled 0 to num_clusters-1.
    """
    clustering = SpectralClustering(n_clusters=num_clusters, random_state=seed).fit(positions)
    clusters = clustering.labels_ 
    cluster_rand = np.random.binomial(1, p, num_clusters)
    D = np.array([cluster_rand[clusters[i]] for i in range(positions.shape[0])])
    return D, clusters

def bernoulli_cluster(num,p,clusters):
  '''
  num (int): number of clusters (should be a perfect square nc*nc)
  p (float): treatment probability in (0,1)
  clusters (n by 1 numpy array): clusters[i] = j says unit i in [n] is in cluster j

  z (numpy array): i-th element is treatment assignment of unit i
  Cz (numpy array): (s,t)-th element is treatment assignment of cluster (s,t)
  flatCz (numpy array): given cluster label k in [nc*nc], return treatment assignment
  '''
  nc = int(np.sqrt(9))
  Cz = (np.random.rand(nc,nc) < p) + 0 # matrix where (s,t) entry is treatment assignment of cluster (s,t)
  flatCz = Cz.flatten() # each cluster (s,t) gets assigned to an index i in [c] where c = number of clusters = (rows+1)*(cols+1)
  treated_cluster_indices = np.where(flatCz == 1)[0] # returns the index labels of clusters that are assigned to treatment
  z = np.isin(clusters,treated_cluster_indices)+0 # if 
  return Cz, flatCz, z

'''
Estimators
SNIPE_linear
SNIPE_beta
cluster_neighbors
CRD_SNIPE_linear
CRD_SNIPE_beta
horvitz_thompson
hajek
diff_in_means_naive
diff_in_means_fraction
'''
def SNIPE_linear(n, p, y, A, z):
    '''
    Returns an estimate of the TTE using our proposed estimator under unit RD

    n (int): number of individuals
    p (float): treatment probability
    y (numpy array?): observations
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    zz = z/p - (1-z)/(1-p)
    return 1/n * y.dot(A.dot(zz))

def SNIPE_beta(n, p, y, A, z, beta):
    treated_neighb = A.dot(z)
    control_neighb = A.dot(1-z)
    est = 0
    for i in range(n):
        w = 0
        a_lim = min(beta,int(treated_neighb[i]))
        for a in range(a_lim+1):
            b_lim = min(beta - a,int(control_neighb[i]))
            for b in range(b_lim+1):
                w = w + ((1-p)**(a+b) - (-p)**(a+b)) * p**(-a) * (p-1)**(-b) * special.binom(treated_neighb[i],a)  * special.binom(control_neighb[i],b)
        est = est + y[i]*w
    return est/n

def cluster_neighborhood(A,i,k):
    '''
    Returns a list of tuples corresponding to the labels (s,t) of clusters adjacent to i
    A = adjacency matrix for population size n^2
    i = the unit we want to compute a neighborhood for
    k = "typical" cluster side length
    '''
    pop_size = np.shape(A)[0] 
    n = int(np.sqrt(pop_size))  # population size is n^2
    nc = int(np.ceil(n/k)**2)   # number of clusters is nc^2

    # get indicies of i's neighbors (nonzero entries in i-th row of A)
    neighbors = np.flatnonzero(A[i,:])

    # We have nc^2 clusters represented by an nc x nc grid
    # We have labels (s,t) in [nc] x [nc] for each cluster
    # We also have labels k in [nc^2] for each cluster
    # Given (s,t), k = nc*s + t. Given k, (s,t)=(np.floor(k/nc),k%nc).
    # For each neighbor, get their cluster assignment (s,t)
    cluster_assignments = []
    for x in np.nditer(neighbors):
        # get the (i,j) coordinate of this neighbor on the population lattice [n] x [n]
        i = int(np.floor(x/n))
        j = x % n
        s,t = lat_toCluster(i,j,k)
        cluster_assignments.append((s,t))
    
    # remove duplicates
    cluster_assignments = list(set(cluster_assignments))

    return cluster_assignments

def CRD_SNIPE_linear():
    pass

def CRD_SNIPE_beta():
    pass

def horvitz_thompson(n, p, y, A, z, clusters=np.array([])):
    '''
    TODO
    '''
    if clusters.size == 0:
        zz = np.prod(np.tile(z/p,(n,1)),axis=1, where=A==1) - np.prod(np.tile((1-z)/(1-p),(n,1)),axis=1, where=A==1)
    else:
        deg = np.sum(clusters,axis=1)
        wt_T = np.power(p,deg)
        wt_C = np.power(1-p,deg)
        zz = np.multiply(np.prod(A*z,axis=1),wt_T) - np.multiply(np.prod(A*(1-z),axis=1),wt_C)
    return 1/n * y.dot(zz)

def hajek(n, p, y, A, z, clusters=np.array([])): 
    '''
    TODO
    '''
    if clusters.size == 0:
        zz_T = np.prod(np.tile(z/p,(n,1)), axis=1, where=A==1)
        zz_C = np.prod(np.tile((1-z)/(1-p),(n,1)), axis=1, where=A==1)
    else:
        deg = np.sum(clusters,axis=1)
        wt_T = np.power(p,deg)
        wt_C = np.power(1-p,deg)
        zz_T = np.multiply(np.prod(A*z,axis=1),wt_T) 
        zz_C = np.multiply(np.prod(A*(1-z),axis=1),wt_C)
    all_ones = np.ones(n)
    est_T = 0
    est_C=0
    if all_ones.dot(zz_T) > 0:
        est_T = y.dot(zz_T) / all_ones.dot(zz_T)
    if all_ones.dot(zz_C) > 0:
        est_C = y.dot(zz_C) / all_ones.dot(zz_C)
    return est_T - est_C

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