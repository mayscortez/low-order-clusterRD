import numpy as np
import random
import networkx as nx
import scipy.sparse

########################################
# Functions to generate random networks
########################################

def erdos_renyi(n,p,undirected=False):
    '''
    Generates a random network of n nodes using the Erdos-Renyi method,
    where the probability that an edge exists between two nodes is p.

    Returns the adjacency matrix of the network as an n by n numpy array
    '''
    A = np.random.rand(n,n)
    A = (A < p) + 0
    A[range(n),range(n)] = 1   # everyone is affected by their own treatment
    if undirected:
        A = symmetrizeGraph(A)
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

def symmetrizeGraph(A):
    n = A.shape[0]
    if A.shape[1] != n:
        print("Error: adjacency matrix is not square!")
        return A
    for i in range(n):
        for j in range(i):
            A[i,j] = A[j,i]
    return A

def lattice1D():
    pass

def lattice2D():
    pass

def lattice3D():
    pass

def rand_points_2D():
    # randomly place points on a 2D plane... randomly add edges???
    pass
########################################
# Functions to generate network weights
########################################

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

    # out_deg = np.array(A.sum(axis=0)).flatten() # array of the out-degree of each node
    # out_deg[out_deg==0] = 1
    # temp = scipy.sparse.diags(C_offdiag/out_deg)
    # C = A.dot(temp)

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

def weights_im_expo(n, d=1, lam=1.5):
    '''
    Returns weights C (numpy array) under the influence and malleability framework,
    with noise distributed as Expo(lam)

    n (int): number of individuals
    d (int): number of influence dimensions
    lam (float): the mean of the Exponential distribution
    '''
    X = np.random.rand(d,n)              # influence
    W = np.random.rand(d,n)              # malliability
    E = np.random.exponential(lam, size=(n,n))
    C = X.T.dot(W)+E
    return C

def weights_im_unif(n, d=1):
    '''
    Returns weights C (numpy array) under the influence and malleability framework,
    with noise distributed as Uniform(0,1)

    n (int): number of individuals
    d (int): number of influence dimensions
    '''
    X = np.random.rand(d,n)              # influence
    W = np.random.rand(d,n)              # malliability
    E = np.random.rand(n,n)
    C = X.T.dot(W)+E
    return C


def weights_node_deg_expo(A, d=1, lam=1, prop=1):
    '''
    Returns weighted adjacency matrix C (numpy array) where weights depend on
    node-degree as Expo(node_deg) or Expo(1/node_deg) or Expo(0)

    A (square numpy array): adjacency matrix of your network
    d (int): influence/malleability dimension
    lam (float): rate of exponential distribution governing noise
    prop (int): governs the dependence of the weights on the node degrees
        prop = 0: both influence & malleability are inverse proportional to node deg
        prop = 1: both influence & malleability are directly proportional to node deg
        prop = 2: influence is inverse proportional to node deg, malleability directly prop
        prop = 3: influence is directly proportional to node deg, malleability inversely prop
    '''
    n = A.shape[0]
    out_deg = np.sum(A,axis=1) # array of the out-degree of each node
    in_deg = np.sum(A,axis=0)  # array of the in-degree of each node
    M_out = np.max(out_deg)    # max out-degree
    M_in = np.max(in_deg)      # max in-degree

    # ensures we don't pass "infinity" as the scale in the next step
    scaled_out_deg = M_out / out_deg
    scaled_in_deg = M_in / in_deg
    scaled_out_deg[scaled_out_deg > M_out] = 0
    scaled_in_deg[scaled_in_deg > M_in] = 0

    if prop == 0:
        X = np.random.exponential(scale=scaled_out_deg, size=(d,n))
        W = np.random.exponential(scale=scaled_in_deg, size=(d,n))
    elif prop == 1:
        X = np.random.exponential(scale=out_deg, size=(d,n))
        W = np.random.exponential(scale=in_deg, size=(d,n))
    elif prop == 2:
        X = np.random.exponential(scale=scaled_out_deg, size=(d,n))
        W = np.random.exponential(scale=in_deg, size=(d,n))
    else:
        X = np.random.exponential(scale=out_deg, size=(d,n))
        W = np.random.exponential(scale=scaled_in_deg, size=(d,n))

    E = np.random.exponential(lam, size=(n,n))

    return (X.T.dot(W)+E)


def weights_node_deg_unif(A, d=1, prop=1, vals=np.array([0,1])):
    '''
    Returns weighted adjacency matrix C (numpy array) where weights depend on
    node-degree as Uniform(0,node_deg) or Uniform(0,1/node_deg) or Uniform(0,0)

    A (square numpy array): adjacency matrix of your network
    d (int): influence/malleability dimension
    lam (float): rate of exponential distribution governing noise
    prop (int): governs the dependence of the weights on the node degrees
        prop = 0: both influence & malleability are inverse proportional to node deg
        prop = 1: both influence & malleability are directly proportional to node deg
        prop = 2: influence is inverse proportional to node deg, malleability directly prop
        prop = 3: influence is directly proportional to node deg, malleability inversely prop
    vals (numpy array): [low, high] => Noise ~ Uniform[low,high]
    '''
    n = A.shape[0]
    out_deg = np.sum(A,axis=1) # array of the out-degree of each node
    in_deg = np.sum(A,axis=0) # array of the in-degree of each node

    # ensures we don't pass "infinity" as the scale in the next step
    scaled_out_deg = 1 / out_deg
    scaled_in_deg = 1 / in_deg
    scaled_out_deg[scaled_out_deg == np.inf] = 0
    scaled_in_deg[scaled_in_deg == np.inf] = 0

    if prop == 0:
        X = np.random.uniform(low=0, high=scaled_out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=scaled_in_deg, size=(d,n))
    elif prop == 1:
        X = np.random.uniform(low=0, high=out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=in_deg, size=(d,n))
    elif prop == 2:
        X = np.random.uniform(low=0, high=scaled_out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=in_deg, size=(d,n))
    else:
        X = np.random.uniform(low=0, high=out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=scaled_in_deg, size=(d,n))

    E = np.random.uniform(low=vals[0], high=vals[1], size=(n,n))

    return (X.T.dot(W)+E)

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

########################################
# Linear Potential Outcomes Model
########################################
linear_pom = lambda C,alpha, z : C.dot(z) + alpha

#####################################################
# Treatment Assignment Mechanisms (Randomized Design)
#####################################################
bernoulli = lambda n,p : (np.random.rand(n) < p) + 0

def completeRD(n,treat):
    '''
    Returns a treatment vector using complete randomized design

    n (int): number of individuals
    p (float): fraction of individuals you want to be assigned to treatment
    '''
    z = np.zeros(shape=(n,))
    z[0:treat] = np.ones(shape=(treat))
    rng = np.random.default_rng()
    rng.shuffle(z)
    return z

def threenet(A):
    '''
    # TODO:
    '''
    A_sq  = 1*(A.dot(A) > 0)
    A_cubed = 1*(A.dot(A_sq) > 0)
    vert = np.arange(A.shape[0])
    center = []
    while vert.size > 0:
        ind = np.random.randint(vert.size)
        center.append(ind)
        neighb = np.flatnonzero(A_cubed[ind,:])
        np.delete(vert, neighb)
    clusters = np.zeros(A.shape[0], len(center))
    vert = np.arange(A.shape[0])
    for i in range(len(center)):
        ind = center[i]
        clusters[ind,i] = 1
        neighb = np.flatnonzero(A[ind,:])
        add_vert = np.intersect1d(vert, neighb, assume_unique=True)
        for j in add_vert:
            clusters[j,i] = 1
        np.delete(vert, neighb)
    for v in vert:
        neighb_2 = np.flatnonzero(A_sq[v,:])
        cent, comm1, comm2 = np.intersect1d(center, neighb_2, assume_unique=True, return_indices=True)
        if len(cent) > 0:
            clusters[v,comm1[0]] = 1
        else:
            neighb_3 = np.flatnonzero(A_cubed[v,:])
            cent, comm1, comm2 = np.intersect1d(center, neighb_3, assume_unique=True, return_indices=True)
            clusters[v,comm1[0]] = 1
    s=np.sum(clusters,axis=1)
    print(np.amax(s))
    print(np.amin(s))
    return clusters

def lattice_cluster1D():
    pass

def lattice_cluster2D():
    pass

def lattice_cluster3D():
    pass

########################################
# Estimators
########################################

def SNIPE_linear(n, p, y, A, z):
    '''
    Returns an estimate of the TTE using our proposed estimator

    n (int): number of individuals
    p (float): treatment probability
    y (numpy array?): observations
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    zz = z/p - (1-z)/(1-p)
    return 1/n * y.dot(A.dot(zz))

def est_ols(y, A, z):
    '''
    Returns an estimate of the TTE using OLS (regresses over proportion of neighbors treated)
    Uses numpy.linalg.lstsq without the use of the normal equations

    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    n = A.shape[0]
    X = np.ones((n,3))
    X[:,1] = z
    X[:,2] = (A.dot(z) - z) / (np.array(A.sum(axis=1)).flatten()-1+1e-10)

    v = np.linalg.lstsq(X,y,rcond=None)[0] # solve for v in y = Xv
    return v[1]+v[2]

def est_ols_treated(y, A, z):
    '''
    Returns an estimate of the TTE using OLS (regresses over number neighbors treated)
    Uses numpy.linalg.lstsq without the use of the normal equations

    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    n = A.shape[0]
    X = np.ones((n,3))
    X[:,1] = z
    X[:,2] = A.dot(z) - z

    v = np.linalg.lstsq(X,y,rcond=None)[0] # solve for v in y = Xv
    return v[1]+(v[2]*(np.sum(A)-n)/n)

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

#Horvitz-Thompson 
def est_ht(n, p, y, A, z, clusters=np.array([])):
  if clusters.size == 0:
    zz = np.prod(np.tile(z/p,(n,1)),axis=1, where=A==1) - np.prod(np.tile((1-z)/(1-p),(n,1)),axis=1, where=A==1)
  else:
    deg = np.sum(clusters,axis=1)
    wt_T = np.power(p,deg)
    wt_C = np.power(1-p,deg)
    zz = np.multiply(np.prod(A*z,axis=1),wt_T) - np.multiply(np.prod(A*(1-z),axis=1),wt_C)
  return 1/n * y.dot(zz)

#Hajek
def est_hajek(n, p, y, A, z, clusters=np.array([])): 
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

def CRD_SNIPE_linear_old(n, p, y, A, z, clusters=np.array([])):
    '''
    TODO: look at this and see if its useful at all... if not delete

    n (int): number of individuals
    p (float): treatment probability
    y (TODO): TODO
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    cluster (numpy array): TODO
    '''
    if clusters.size == 0:
        z_c = z
        A_c = A
    else:
        z_c = 1*(np.sum(np.multiply(z.reshape((n,1)),clusters),axis=0)>0)
        A_c = 1*(A.dot(clusters) >0)
    zz = z_c/p - (1-z_c)/(1-p)
    return 1/n * y.dot(A_c.dot(zz))

def CRD_SNIPE_linear():
    pass

def CRD_SNIPE_naive_linear():
    pass