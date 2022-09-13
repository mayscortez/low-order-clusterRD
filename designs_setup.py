import numpy as np

#####################################################
# Clustering
#####################################################
def lattice_cluster1D():
  pass

def lattice_cluster2D():
  pass

def square_cluster(k,L):
  '''
  Clustering of a k by k lattice graph where each cluster is an L by L grid graph

  k (int): indicates we want to cluster a k by k lattice graph
  L (int): L should be a positive integer dividing k i.e. k%L=0
  
  We end up with c = (k*k)/(L*L) clusters where each cluster is an L by L lattice

  clusters (numpy array): cluster[i] = j indicates the i-th individual belongs to cluster j
  for i=0,1,...,n-1 and j=0,1,...,c-1 where n = k*k is the total number of nodes in the graph
  '''
  a = k/L           # number of clusters per row or column
  c = k**2//L**2     # total number of clusters
  C = np.zeros((k,k)) # represents the nodes of a k by k lattice graph
  row_start = 0     # 
  row_end = L
  col_start = 0
  col_end = L

  for i in range(1,c):
    if (i%a == 0):
      row_start = row_start + L
      row_end = row_end + L
      col_start = 0
      col_end = L
    else:
      col_start = col_start + L
      col_end = col_end + L
    C[row_start:row_end, col_start:col_end] = i
  
  clusters = C.flatten()
  return clusters

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

def one_hop_max():
  pass

def lattice_cluster3D():
    pass

#####################################################
# Cluster Randomized Designs
#####################################################
def bernoulli_cluster(c,p,clusters):
  '''
  c (int): number of clusters
  p (float): treatment probability in (0,1)
  clusters (numpy array): clusters[i] = j says unit i is in cluster j

  z (numpy array): i-th element is treatment assignment of unit i
  Cz (numpy array): j-th element is treatment assignment of cluster j
  '''
  Cz = (np.random.rand(c) < p) + 0
  treated_cluster_indices = np.where(Cz == 1)[0]
  z = np.isin(clusters,treated_cluster_indices)+0
  return Cz, z

def stratified_cluster():
  pass

#####################################################
# Unit Randomized Designs
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

def seq_treatment_probs(M, p):
  '''
  Returns sequence of treatment probabilities for Bernoulli staggered rollout

  M (int): fineness of measurements in staggered rollout (# timesteps - 1, not counting the time zero)
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05
  '''
  fun = lambda i: (i)*p/(M)
  P = np.fromfunction(fun, shape=(M+1,))
  return P

def seq_treated(M, p, n, K=np.array([])):
  '''
  Returns number of people treated by each time step with K = [k0, k1, ... , kM] via ki = i*n*p/M
  
  M (int): fineness of measurements in staggered rollout (# timesteps - 1, not counting the time zero)
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05
  n (int): size of population
  '''
  if K.size == 0:
    fun = lambda i: np.floor(p*n*i/M).astype(int)
    K = np.fromfunction(fun, shape=(M+1,))
  return K

def staggered_rollout_complete(n, K):
  '''
  Returns Treatment Samples Z from Complete Staggered Rollout and number of people treated by each time step K

  beta (int): degree of potential outcomes model
  n (int): size of population
  K (numpy array): total number of individuals treated by each timestep
  '''

  ### Initialize ###
  Z = np.zeros(shape=(K.size,n))   # for each treatment sample, z_t
  indices = np.random.permutation(np.arange(n))           # random permutation of the individuals

  ### staggered rollout experiment ###
  # indices: holds indices of entries equal to 0 in treatment vector
  # to_treat: from the next set of indiv in the random permutation
  for t in range(K.size-1):
    to_treat = indices[K[t]:K[t+1]+1]
    Z[t+1:,to_treat] = 1 

  return Z

def complete_coeffs(n, K):
  '''
  Returns coefficients l_t from Complete Staggered Rollout

  n (int): size of population
  K (numpy array): total number of individuals treated by each timestep
  '''

  ### Initialize ###
  L = np.zeros(K.size)             # for the coefficients L_t

  for t in range(K.size):
    n_minusK = n - K            # [n-k0, n-k1, ... , n-k_beta]
    kt_minusK = K[t] - K        # [kt-k0, kt-k1, ... , kt-k_beta]
    minusK = -1*K               # [-k0, -k1, ... , -k_beta]
    n_minusK[t] = 1; kt_minusK[t] = 1; minusK[t] = 1
    fraction1 = n_minusK/kt_minusK
    fraction2 = minusK/kt_minusK
    L[t] = np.prod(fraction1) - np.prod(fraction2)

  return L

def outcome_sums(Y, Z):
  '''
  Returns the sums of the outcomes Y(z_t) for each timestep t

  Y (function): potential outcomes model
  Z (numpy array): treatment vectors z_t for each timestep t
   - each row should correspond to a timestep, i.e. Z should be beta+1 by n
  '''
  sums = np.zeros(Z.shape[0]) 
  for t in range(Z.shape[0]):
    sums[t] = np.sum(Y(Z[t,:]))
  return sums