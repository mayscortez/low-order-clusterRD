from scipy.sparse import csr_array
from myFunctions import *
import numpy as np 

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
    neighborhoods = [list(row.nonzero()[1]) for row in A] # list of neighbors of each unit
    neighborhood_sizes = A.sum(axis=1).tolist() # size of each unit's neighborhood
    neighbor_treatments = [list(z[neighborhood]) for neighborhood in neighborhoods] # list of treatment assignments in each neighborhood

    A = A.multiply(csr_array(np.tile(np.repeat(np.arange(1,nc+1),n//nc), (n,1)))) # modifies the adjancecy matrix so that if there's an edge from j to i, A[i,j]=cluster(j)
    cluster_neighborhoods = [np.unique(row.data,return_counts=True) for row in A] # for each i, cluster_neighborhoods[i] = [a list of clusters i's neighbors belong to, a list of how many neighbors are in each of these clusters]
    cluster_neighborhood_sizes = [len(x[0]) for x in cluster_neighborhoods] # size of each unit's cluster neighborhood
    
    # Probabilities of each person's neighborhood being entirely treated or entirely untreated
    all_treated_prob = np.multiply(np.power(p, neighborhood_sizes), np.power(q, cluster_neighborhood_sizes))
    none_treated_prob = [np.prod((1-q) + np.power(1-p, x[1])*q) for x in cluster_neighborhoods]
    
    # Indicators of each person's neighborhood being entirely treated or entirely untreated
    all_treated = [np.prod(treatments) for treatments in neighbor_treatments]
    none_treated = [all(z == 0 for z in treatments)+0 for treatments in neighbor_treatments]

    zz = np.nan_to_num(np.divide(all_treated,all_treated_prob) - np.divide(none_treated,none_treated_prob))

    return 1/n * y.dot(zz)

def run_experiment(beta, n, nc, B, r, diag, Pii, Pij, phi, design, q_or_K, T):
    '''
    beta = degree of the model / polynomial
    n = population size
    nc = number of clusters
    B = original treatment budget/fraction
    r = ratio offdiag/diag: (indirect effect)/(direct effects)
    diag = maxium norm of the direct effects before covariate type scaling
    Pii = edge probability within communities
    Pij = edge prob btwn different communities
    phi = correlation btwn community & effect type (probability between 0 and 0.5)
    design = design being used for selecting clusters, either "complete" or "bernoulli"
    q_or_K = if using complete RD for selecting cluster, this will be the value of K; if using Bernoulli design, this will be the value q
    graphNum = number of graphs to average over
    T = number of trials per graph
    graphStr = type of graph e.g. "SBM" for stochastic block model or "ER" for Erdos-Renyi
    p_prime = the budget on the boundary of U
    '''
    
    offdiag = r*diag   # maximum norm of indirect effect

    if design == "complete":
        K = q_or_K
        q = K/nc
    else:
        q = q_or_K
        K = int(np.floor(q*nc))

    p = B/q
    G, A = SBM(n, nc, Pii, Pij)  #returns the SBM networkx graph object G and the corresponding adjacency matrix A

    # random weights for the graph edges
    rand_wts = np.random.rand(n,3)
    alpha = rand_wts[:,0].flatten()
    C = simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
    C = covariate_weights_binary(C, minimal = 1/4, extreme = 4, phi=phi)
    
    # potential outcomes model
    fy = ppom(beta, C, alpha)

    # compute the true TTE
    TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
    print("TTE: {}\n".format(TTE))
    
    ####### Estimate ########

    # parameters for the staggered rollout - Cluster Randomized Design
    P = seq_treatment_probs(beta, p)        # treatment probabilities for each step of the staggered rollout on U
    P_prime = seq_treatment_probs(beta, 0)  # treatment probabilities for each step of the staggered rollout on the boundary of U

    TTE_ht = np.zeros(T)
    for i in range(T):
        # select clusters 
        if design == "complete":
            selected = select_clusters_complete(nc, K)
        else:
            selected = select_clusters_bernoulli(nc, q)
        
        # U
        selected_nodes = [x for x,y in G.nodes(data=True) if (y['block'] in selected)] # get the nodes in selected clusters

        # Cluster Randomized Design
        Z = staggered_rollout_bern_clusters(n, selected_nodes, P, [], P_prime)
        z = Z[beta,:]
        y = fy(z)

        TTE_ht[i] = horvitz_thompson(n, nc, y, A, z, q, p)

    print("H-T: {}".format(np.sum(TTE_ht)/T))
    print("H-T relative bias: {}".format(((np.sum(TTE_ht)/T)-TTE)/TTE)) #(est-TTE)/TTE ((np.sum(TTE_ht)/T)-TTE)/TTE
    print("H-T MSE: {}\n".format(np.sum((TTE_ht-TTE)**2)/T))

def covariate_weights_binary(C, minimal = 1/4, extreme = 4, phi=0):
    '''
    Returns a weighted adjacency matrix where weights are determined by covariate type. We assume a binary effect types covariate

    C (array): weights without effect type covariate
    minimal (float): 
    extreme (int):
    phi: probability that an individual's covariate type flips
    '''
    n = C.shape[0]
    scaling1 = np.zeros(n//2) + minimal
    scaling2 = np.zeros(n//2) + extreme

    R1 = np.random.rand(n//2)
    R2 = np.random.rand(n//2)
    R1 = (R1 < phi) + 0
    R2 = (R2 < phi) + 0

    scaling1[np.nonzero(R1)] = extreme
    scaling2[np.nonzero(R2)] = minimal

    scaling = np.concatenate((scaling1,scaling2))
    return C.multiply(scaling)

def SBM(n, k, Pii, Pij):
    '''
    Returns the adjacency matrix (as a scipy sparse array) of a stochastic block model on n nodes with k communities
    The edge prob within the same community is Pii
    The edge prob across different communities is Pij
    '''
    sizes = np.zeros(k, dtype=int) + n//k
    probs = np.zeros((k,k)) + Pij
    np.fill_diagonal(probs, Pii)
    G = nx.stochastic_block_model(sizes, probs)
    A = nx.adjacency_matrix(nx.stochastic_block_model(sizes, probs))
    A.setdiag(1)
    #blocks = nx.get_node_attributes(G, "block")
    return G, A

avg_deg = 10
beta = 1
n = 1000
nc = 50
B = 0.5
p = 1
r = 1.25
diag = 1
Pii = avg_deg/(n/nc)
Pij = 0
phi = 0
#design = "complete"
#q_or_K = int(np.floor(B * nc / p))
design = "bernoulli"
q_or_K = 0.5
T = 100

run_experiment(beta, n, nc, B, r, diag, Pii, Pij, phi, design, q_or_K, T)