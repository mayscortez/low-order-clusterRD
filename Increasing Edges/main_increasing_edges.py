'''
How does the bias and variance of the estimator change as we increase the number of edges between clusters?
'''

# Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from myFunctions import *

def main(beta, graphNum, T):
    deg_str = '_deg' + str(beta)  # naming convention
    save_path = 'output/' + 'deg' + str(beta) + '/'                    
    
    graphStr = "SBM"    # stochastic block model    
    n = 1000            # number of nodes
    nc = 50             # number of communities
    Pii = 10/(n/nc)     # edge probability within communities

    K = int(nc/2)           # number of clusters to be in experiment if choosing via complete RD
    q = 0.5            # fraction of clusters to be part of the experiment if choosing via Bernoulli RD
    B = 0.5            # original treatment budget
    p_prime = 0        # fraction of the boundary of U to get treated as well
    RD = "complete"    # either "complete" or "bernoulli" depending on the design used for selecting clusters
    
    if RD == "complete":
        q_or_K = K
        q_or_K_st = '_K' + str(K)
    else:
        q_or_K = q
        q_or_K_st = '_q' + str(q).replace('.','')
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'B' + str(B).replace('.','') + q_or_K_st  # naming convention
    
    f = open(save_path +'incr_edges' + fixed + deg_str + '.txt', 'w') # e.g filename could be incr_edges_n10000_nc4_B025_K1_deg2.txt  
    
    startTime1 = time.time()
    ##################################################################
    # Run Experiment: Increasing the number of edges crossing clusters
    ##################################################################
    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    
    results = []
    #Pij = np.array([0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]) #0.005, 0.01])
    p_ins = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2] # np.linspace(Pii,Pii/2,6)
    for p_in in p_ins:
        p_out = (0.5-p_in)/49
        p_out = np.around(p_out, 5)
        print("p_in = {} and p_out = {}".format(p_in, p_out))
        startTime2 = time.time()

        results.extend(run_experiment(beta, n, nc, B, r, diag, p_in, p_out, RD, q_or_K, p_prime, graphNum, T, graphStr))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for (p_in,p_out) = ({},{}) step: {}'.format(p_in, p_out,executionTime),file=f)
        print('Runtime (in seconds) for (p_in,p_out) = ({},{}) step: {}'.format(p_in, p_out,executionTime))

    executionTime = (time.time() - startTime1)
    print('Total runtime in minutes: {}'.format(executionTime/60),file=f)   
    print('Total runtime in minutes: {}'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + graphStr + fixed + '_incrEdges-full-data'+deg_str +'.csv')
    


def run_experiment(beta, n, nc, B, r, diag, Pii, Pij, design, q_or_K, p_prime, graphNum, T, graphStr):
    '''
    n = population size
    nc = number of clusters
    B = original treatment budget/fraction
    r = ratio offdiag/diag (indirect effect)/(direct effects)
    diag = maxium norm of the direct effects
    Pii = edge probability within communities
    Pij = edge prob between communities
    design = either "complete" or "bernoulli" depending on which design is being used for selecting clusters
    q_or_K = if using complete RD for selecting cluster, this will be the value of K; if using Bernoulli design, this will be the value q
    p_prime = the budget on the boundary of U
    graphNum = number of graphs to average over
    T = number of trials per graph
    graphStr = type of graph e.g. "SBM" for stochastic block model or "ER" for Erdos-Renyi
    '''
    
    offdiag = r*diag   # maximum norm of indirect effect

    if design == "complete":
        K = q_or_K
        q = K/nc
    else:
        q = q_or_K
        K = int(np.floor(q*nc))

    p = B/q
    unitsPercluster = n/nc
    edges_out_over_in = (Pij*(K-1)*unitsPercluster) / (Pii*K*unitsPercluster) # ratio of expected number of edges within cluster : outside cluster for each unit
    dict_base = {'n': n, 'nc': nc, 'Pii': Pii, 'Pij': Pij, 'K': K, 'p': p, 'q': q, 'B': B, 'ratio': r, 'out-in': edges_out_over_in}

    G, A = SBM(n, nc, Pii, Pij)  #returns the SBM networkx graph object G and the corresponding adjacency matrix A

    G_transitivity = nx.transitivity(G)
    G_avgClusteringCoeff = nx.average_clustering(G)
    dict_base.update({'global': G_transitivity, 'average': G_avgClusteringCoeff})

    # plot_degrees(G, Pii, Pij)
    # draw_SBM(n, K, G, Pii, Pij)

    results = []
    for g in range(graphNum):
        graph_rep = str(g)
        dict_base.update({'Graph':graphStr+graph_rep})

        # random weights for the graph edges
        rand_wts = np.random.rand(n,3)
        alpha = rand_wts[:,0].flatten()
        C = simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
        
        # potential outcomes model
        fy = ppom(beta, C, alpha)

        # compute the true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        
        P = seq_treatment_probs(beta, p)    # treatment probabilities for each step of the staggered rollout
        P_prime = seq_treatment_probs(beta, p_prime)    # treatment probabilities for each step of the staggered rollout
        H = bern_coeffs(P)                  # coefficients for the polynomial interpolation estimator

        ####### Estimate ########
        #TODO: Add estimator that only looks at U
        estimators = []
        estimators.append(lambda y,z,sums,H_m: graph_agnostic(n,sums,H_m)/q)
        estimators.append(lambda y,z, sums, H_m: poly_regression_prop(beta, y,A,z))
        estimators.append(lambda y,z, sums, H_m: poly_regression_num(beta, y,A,z))
        estimators.append(lambda y,z,sums,H_m: diff_in_means_naive(y,z))
        estimators.append(lambda y,z,sums,H_m: diff_in_means_fraction(n,y,A,z,0.75))
        num_of_estimators = 5

        alg_names = ['PI($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
        #alg_names = ['PI($p$)', 'newname' ,'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']

        for i in range(T):
            selected = select_clusters_complete(nc, K) # select clusters 
            selected_nodes = [x for x,y in G.nodes(data=True) if (y['block'] in selected)] # get the nodes in selected clusters
            boundary_of_selected = [] #TODO get boundary

            dict_base.update({'rep': i})

            #TODO: compute these for just the selected nodes as well
            Z = staggered_rollout_bern(n, selected_nodes, P, boundary_of_selected, P_prime)
            z = Z[beta,:]
            y = fy(z)
            sums = outcome_sums(fy, Z)

            for x in range(num_of_estimators):
                est = estimators[x](y,z,sums,H) # have it include both the parameters for all as well as just U
                dict_base.update({'Estimator': alg_names[x], 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Bias_sq': ((est-TTE)**2)})
                results.append(dict_base.copy())

    return results

def SBM(n, k, Pii, Pij):
    '''
    Returns the adjacency matrix of a stochastic block model on n nodes with k communities
    The edge prob within the same community is Pii
    The edge prob across different communities is Pij
    '''
    sizes = np.zeros(k, dtype=int) + n//k
    probs = np.zeros((k,k)) + Pij
    np.fill_diagonal(probs, Pii)
    G = nx.stochastic_block_model(sizes, probs)
    A = nx.adjacency_matrix(nx.stochastic_block_model(sizes, probs))
    #blocks = nx.get_node_attributes(G, "block")
    return G, A


def color_nodes(n, K):
    '''
    Returns random colors for each of the K communities in a SBM

    # https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    int n = number of nodes
    int K = number of communities (assumed to be of equal size, with n divisible  by K)
    '''
    number_of_colors = K
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    rep = int(n/K)
    res = np.repeat(color, rep)
    return res
        
def plot_degrees(G, Pii, Pij):
    '''
    Saves a degree histogram for a networkx SBM. 

    G (networkx graph object): the SBM
    Pii (float): probability of edge between two nodes of the same community
    Pij (float): probabiility of an edge between two nodes of different communities
    '''
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    fig, ax2 = plt.subplots()
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    Pin = str(Pii).replace('.','')
    Pout = str(Pij).replace('.','')
    plt.savefig('testing-graphs/degrees/degrees-' + 'Pin' + Pin + '-Pout' + Pout + '.png')
    plt.close()

def draw_SBM(n, K, G, Pii, Pij):
    '''
    Saves a drawing of a networkx SBM.

    n (int): number of nodes
    K (int): number of communities
    G (networkx graph object): SBM
    Pii (float): probability of edge between two nodes of the same community
    Pij (float): probabiility of an edge between two nodes of different communities
    '''
    fig2, ax1 = plt.subplots()
    colors = color_nodes(n,K)
    nx.draw_networkx(G, ax=ax1, with_labels=False, node_size=10, node_color=colors,width=0.4)
    ax1.set_title("SBM with $P_{{{}}}={}$ and $P_{{{}}}={}$".format('in', Pii, 'out', Pij))
    fig2.tight_layout()
    Pin = str(Pii).replace('.','')
    Pout = str(Pij).replace('.','')
    plt.savefig('testing-graphs/structure/structure-' + 'Pin' + Pin + '-Pout' + Pout + '.png')
    pass