'''
How does the bias and variance of the estimator change as we increase the number of edges between clusters?
'''

# Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from myFunctions_edges import *

def main(beta, graphNum, T, B=0.06, p=1, cluster_selection_RD = "bernoulli"):
    deg_str = '_deg' + str(beta)  # naming convention
    save_path = 'output/' + 'deg' + str(beta) + '/'                    
    
    graphStr = "SBM"    # stochastic block model    
    n = 1000            # number of nodes
    nc = 50             # number of communities

    K_expected = int(B * nc / p)   # if complete design, this is actual number of chosen clusters; if Bernoulli this is expected number of chosen clusters
    q_expected = K_expected/nc 
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'B' + str(B).replace('.','') + '_p' + str(np.round(p,3)).replace('.','')  # naming convention
    
    f = open(save_path +'incr_edges' + fixed + deg_str + '_' + cluster_selection_RD + '.txt', 'w') # e.g filename could be incr_edges_n10000_nc4_B025_K1_deg2.txt  
    
    startTime1 = time.time()
    ##################################################################
    # Run Experiment: Increasing the number of edges crossing clusters
    ##################################################################
    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    
    results = []
    #Pij = np.array([0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]) #0.005, 0.01])
    p_ins = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.01] # np.linspace(Pii,Pii/2,6)
    for p_in in p_ins:
        p_out = (0.5-p_in)/49
        p_out = np.around(p_out, 5)
        print("p_in = {} and p_out = {}".format(p_in, p_out))
        startTime2 = time.time()

        results.extend(run_experiment(beta, n, nc, B, p, r, diag, p_in, p_out, cluster_selection_RD, q_expected, K_expected, graphNum, T, graphStr))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for (p_in,p_out) = ({},{}) step: {}'.format(p_in, p_out,executionTime),file=f)
        print('Runtime (in seconds) for (p_in,p_out) = ({},{}) step: {}'.format(p_in, p_out,executionTime))

    executionTime = (time.time() - startTime1)
    print('Degree: {}'.format(beta)) 
    print('Total runtime in minutes: {}'.format(executionTime/60),file=f)   
    print('Total runtime in minutes: {}'.format(executionTime/60))     
    print('')    
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + 'incrEdges' + fixed + deg_str + '_' + cluster_selection_RD + '-full.csv')
    


def run_experiment(beta, n, nc, B, target_p, r, diag, Pii, Pij, design, Eq, EK, graphNum, T, graphStr, p_prime=0, realized=True):
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

    unitsPercluster = n/nc
    edges_out_over_in = (Pij*(EK-1)*unitsPercluster) / (Pii*EK*unitsPercluster) # ratio of expected number of edges within cluster : outside cluster for each unit
    dict_base = {'n': n, 'nc': nc, 'Pii': Pii, 'Pij': Pij, 'B': B, 'p': target_p, 'EK': EK, 'Eq': Eq, 'ratio': r, 'out-in': edges_out_over_in}

    # Cluster Randomized Design Estimators
    estimators_clusterRD = []
    estimators_clusterRD.append(lambda q,y,z,sums,H_m,sums_U: PI(n*q,sums,H_m))             # estimator looks at all [n]
    estimators_clusterRD.append(lambda q,y,z,sums,H_m,sums_U: PI(n*q,sums_U,H_m))           # estimator only looking at [U]
    estimators_clusterRD.append(lambda q,y,z,sums,H_m,sums_U: horvitz_thompson(n, nc, y, A, z, Eq, target_p))  
    estimators_clusterRD.append(lambda q,y,z,sums,H_m,sums_U: DM_naive(y,z))                 # difference in means 
    estimators_clusterRD.append(lambda q,y,z,sums,H_m,sums_U: DM_fraction(n,y,A,z,0.75))     # thresholded difference in means

    alg_names_clusterRD = ['PI-$n$($p$)', 'PI-$\mathcal{U}$($p$)', 'HT', 'DM-C', 'DM-C($0.75$)']

    # Bernoulli Randomized Design Estimators
    estimators_bernRD = []
    estimators_bernRD.append(lambda y,z,sums,H_m: PI(n,sums,H_m))
    estimators_bernRD.append(lambda y,z, sums, H_m: poly_LS_prop(beta, y,A,z))      # polynomial regression
    estimators_bernRD.append(lambda y,z, sums, H_m: poly_LS_num(beta, y,A,z))
    estimators_bernRD.append(lambda y,z,sums,H_m: DM_naive(y,z))                 # difference in means 
    estimators_bernRD.append(lambda y,z,sums,H_m: DM_fraction(n,y,A,z,0.75))     # thresholded difference in means

    alg_names_bernRD = ['PI-$n$($B$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']

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
        
        ####### Estimate ########

        # parameters for the staggered rollout - Cluster Randomized Design
        P = seq_treatment_probs(beta, target_p)        # treatment probabilities for each step of the staggered rollout on U
        P_prime = seq_treatment_probs(beta, 0)  # treatment probabilities for each step of the staggered rollout on the boundary of U
        H = bern_coeffs(P)                      # coefficients for the polynomial interpolation estimator

        # parameters for the staggered rollout - Bernoulli Randomized Design
        P_bernRD = seq_treatment_probs(beta, B) # treatment probabilities for each step of the staggered rollout on [n]
        H_bern = bern_coeffs(P_bernRD)          # coefficients for the polynomial interpolation estimator

        for i in range(T):
            # select clusters 
            if design == "complete":
                selected = select_clusters_complete(nc, EK)
                K_real = EK
                q_real = Eq
            else:
                selected = select_clusters_bernoulli(nc, Eq)
                K_real = len(selected)
                q_real = K_real/nc

            selected_nodes = [x for x,y in G.nodes(data=True) if (y['block'] in selected)] # get the nodes in selected clusters

            dict_base.update({'rep': i, 'design': 'Cluster', 'K': K_real, 'q': q_real})

            # Cluster Randomized Design
            Z = staggered_rollout_bern_clusters(n, selected_nodes, P, [], P_prime)
            z = Z[beta,:]
            y = fy(z)
            sums, sums_U = outcome_sums(n, fy, Z, selected_nodes) # the sums corresponding to all nodes (i.e. [n]) and just selected nodes (i.e. [U])

            for x in range(len(estimators_clusterRD)):
                if realized:
                    bias_correction = 1 - (1-Eq)**nc
                    est = estimators_clusterRD[x](q_real*bias_correction,y,z,sums,H,sums_U)
                else:
                    est = estimators_clusterRD[x](Eq,y,z,sums,H,sums_U)
                dict_base.update({'Estimator': alg_names_clusterRD[x], 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Rel_bias_sq':((est-TTE)/TTE)**2, 'Bias_sq': ((est-TTE)**2)})
                results.append(dict_base.copy())

            # Bernoulli Randomized Design (No Clusters)
            dict_base.update({'design': 'Bernoulli'})
            Z_bern = staggered_rollout_bern(n, P_bernRD)
            z_bern = Z_bern[beta,:]
            y_bern = fy(z)
            sums_bern = outcome_sums(n, fy, Z_bern, range(0,n))[0]

            for x in range(len(estimators_bernRD)):
                est = estimators_bernRD[x](y_bern,z_bern,sums_bern,H_bern) # have it include both the parameters for all as well as just U
                dict_base.update({'Estimator': alg_names_bernRD[x], 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Rel_bias_sq':((est-TTE)/TTE)**2, 'Bias_sq': ((est-TTE)**2)})
                results.append(dict_base.copy())

    return results

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