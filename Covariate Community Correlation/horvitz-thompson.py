from scipy.sparse import csr_array
import experiment_functions as exFun
import numpy as np 
import time
import warnings
warnings.simplefilter('ignore')

def horvitz_thompson(n, nc, y, A, z, q, p):
    '''Computes the Horvitz-Thompson estimate of the TTE under Cluster-Bernoulli design.
    
    Parameters
    ----------
    n : int
        the size of the population/network
    nc : int
        the number of clusters
    y : numpy array
        the outcomes of each unit in the population
    A : scipy sparse array
        adjacency matrix of the network such that A[i,j]=1 indicates that unit j is an in-neighbor of i
    z : numpy array
        the treatment assignment of each unit in the population
    q : float
        probability that a cluster is indepdently chosen for treatment
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
    G, A = exFun.SBM(n, nc, Pii, Pij)  #returns the SBM networkx graph object G and the corresponding adjacency matrix A

    # random weights for the graph edges
    rand_wts = np.random.rand(n,3)
    alpha = rand_wts[:,0].flatten()
    C = exFun.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
    C = exFun.covariate_weights_binary(C, minimal = 1/4, extreme = 4, phi=phi)
    
    # potential outcomes model
    fy = exFun.ppom(beta, C, alpha)

    # compute the true TTE
    TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
    print("TTE: {}\n".format(TTE))
    
    ####### Estimate ########

    # parameters for the staggered rollout - Cluster Randomized Design
    P = exFun.seq_treatment_probs(beta, p)        # treatment probabilities for each step of the staggered rollout on U
    P_prime = exFun.seq_treatment_probs(beta, 0)  # treatment probabilities for each step of the staggered rollout on the boundary of U

    TTE_ht = np.zeros(T)
    for i in range(T):
        # select clusters 
        if design == "complete":
            selected = exFun.select_clusters_complete(nc, K)
        else:
            selected = exFun.select_clusters_bernoulli(nc, q)
        
        # U
        selected_nodes = [x for x,y in G.nodes(data=True) if (y['block'] in selected)] # get the nodes in selected clusters

        # Cluster Randomized Design
        Z = exFun.staggered_rollout_bern_clusters(n, selected_nodes, P, [], P_prime)
        z = Z[beta,:]
        y = fy(z)

        TTE_ht[i] = horvitz_thompson(n, nc, y, A, z, q, p)

    print("H-T: {}".format(np.sum(TTE_ht)/T))
    print("H-T relative bias: {}".format(((np.sum(TTE_ht)/T)-TTE)/TTE)) #(est-TTE)/TTE ((np.sum(TTE_ht)/T)-TTE)/TTE
    print("H-T MSE: {}\n".format(np.sum((TTE_ht-TTE)**2)/T))
 

########################
## Run the experiment ##
########################
avg_deg = 10            # average network degree
beta = 1                # model degree
n = 1000                # network size
nc = 50                 # number of clusters/blocks
B = 0.5                 # treatment budget
p = 1                   # treatment probability for those in chosen clusters
r = 1.25                # governs the relative magnitude of the indirect effects relative to the direct effects
diag = 1                # maxium norm of the direct effects before covariate type scaling
Pii = avg_deg/(n/nc)    # edge probability between two nodes in the same cluster
Pij = 0                 # edge probability between two nodes in different clusters
phi = 0                 # probability of switching covariate types (phi = 0 means perfect homophily, phi = 0.5 means no homophily)
design = "bernoulli"    # how to choose clusters; alternatively, design = "complete"
q_or_K = 0.5            # q_or_K = int(np.floor(B * nc / p))
T = 100                 # number of treatment assignment samples to average over

startTime = time.time()
run_experiment(beta, n, nc, B, r, diag, Pii, Pij, phi, design, q_or_K, T)
executionTime = time.time()-startTime
print('Total runtime in seconds: {}\nTotal runtime in minutes: {}'.format(executionTime, executionTime/60)) 