'''
How does the bias and variance of the estimator change as we reduce the extrapolation error (increase p)?
'''

# Setup
import numpy as np
import pandas as pd
import time
from experiment_functions import PI, horvitz_thompson, DM_naive, DM_fraction, poly_LS_prop, poly_LS_num, SBM, binary_covariate_weights, ppom, seq_treatment_probs, bern_coeffs, select_clusters_complete, select_clusters_bernoulli, staggered_rollout_bern_clusters, outcome_sums, staggered_rollout_bern
import pickle

def main(beta, graphNum, T, B=0.06, phi=0, p_in=0.5, cluster_selection_RD = "bernoulli"):
    '''
    beta (int): degree of the model
    graphNum (int): how many graph models to average over
    T (int): how many treatment samples to average over
    B (float): treatment budget (marginal treatment probability)
    p (float): treatment probability if cluster is chosen (treatment prob conditioned on being in U)
    cluster_selection_RD (str): either "complete" or "bernoulli" depending on the design used for selecting clusters
    '''
    deg_str = '_deg' + str(beta)  # naming convention
    save_path = 'output/' + 'deg' + str(beta) + '/' + cluster_selection_RD + '/' # e.g. save to the folder output/deg1/bernoulli/
    experiment = 'extrapolation'                    
    
    graphStr = "SBM"          # stochastic block model    
    n = 1000                  # number of nodes
    nc = 50                   # number of communities
    p_out = (0.5-p_in)/49     # edge probability between different communities, this way keeps expected degree constant at 10
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_B' + str(B).replace('.','') + '_phi' + str(np.round(phi,3)).replace('.','') # naming convention
    
    f = open(save_path + experiment + fixed + deg_str + '_' + cluster_selection_RD + '.txt', 'w') # e.g filename could be correlation_n1000_nc50_in005_out0_p1_B01_deg2_bernoulli.txt 
    startTime1 = time.time()

    #################################################################
    # Run Experiment: Increasing the correlation between covariates
    #################################################################
    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    
    results = []
    if B == 0.06:
        probs = [3/50, 3/25, 3/18, 3/12, 3/9, 3/7, 3/5, 3/4, 1]  # K in [50, 25, 18, 12, 9, 7, 5, 4, 3]
    elif B == 0.02:
        probs = [1/50, 1/25, 1/18, 1/12, 1/8, 1/4, 1/3, 1/2, 1]  # K in [50, 25, 18, 12, 8, 4, 3, 2, 1] 
    else:
        raise ValueError("Need to set probs in main_extrapolation for this value of B")

    for p in probs:
        print("p = {}".format(p))
        startTime2 = time.time()

        K_expected = int(B * nc / p)   # if complete design, this is actual number of chosen clusters; if Bernoulli this is expected number of chosen clusters
        q_expected = K_expected/nc 

        results.extend(run_experiment(beta, n, nc, B, p, r, diag, p_in, p_out, phi, cluster_selection_RD, q_expected, K_expected, graphNum, T, graphStr))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for p = {}, E[K] = {} step: {}'.format(np.round(p,3), K_expected, executionTime),file=f)
        print('Runtime (in seconds) for p = {}, E[K] = {} step: {}'.format(np.round(p,3), K_expected, executionTime))

    executionTime = (time.time() - startTime1)
    print('Degree: {}'.format(beta)) 
    print('Total runtime in minutes: {}'.format(executionTime/60),file=f)   
    print('Total runtime in minutes: {}'.format(executionTime/60)) 
    print('')       
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + experiment + fixed + deg_str + '_' + cluster_selection_RD + '-full.csv') # e.g filename could be correlation_n1000_nc50_in005_out0_p1_B01_deg2_bernoulli-full.csv 
    

def run_experiment(beta, n, nc, B, target_p, r, diag, Pii, Pij, phi, design, Eq, EK, graphNum, T, graphStr, p_prime=0, realized = True):
    '''
    beta = degree of the model / polynomial
    n = population size
    nc = number of clusters
    B = original treatment budget/fraction
    target_p = the treatment probability within clusters i.e. the target treatment probability - note this may differ from the realized t.p.
    r = ratio offdiag/diag: (indirect effect)/(direct effects)
    diag = maxium norm of the direct effects before covariate type scaling
    Pii = edge probability within communities
    Pij = edge prob btwn different communities
    phi = correlation btwn community & effect type (probability between 0 and 0.5)
    design = design being used for selecting clusters, either "complete" or "bernoulli"
    Eq = expected value of q E[K/nc] (i.e. expected fraction of clusters that are chosen)
    EK = expected value of K E[K] (i.e. expected number of chosen clusters)
    graphNum = number of graphs to average over
    T = number of trials per graph
    graphStr = type of graph e.g. "SBM" for stochastic block model or "ER" for Erdos-Renyi
    p_prime = the budget on the boundary of U
    realized = if True, uses realized value of q in the estimator instead of expected (this only changes things if we use Bernoulli design to choose clusters)
    '''
    
    offdiag = r*diag   # maximum norm of indirect effect

    dict_base = {'n': n, 'nc': nc, 'Pii': Pii, 'Pij': Pij, 'Phi': phi, 'B': B, 'p': target_p, 'EK': EK, 'Eq': Eq, 'ratio': r}

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

    results = []
    for g in range(graphNum):
        graph_rep = str(g)
        dict_base.update({'Graph':graphStr+graph_rep})

        G, A = SBM(n, nc, Pii, Pij)  #returns the SBM networkx graph object G and the corresponding adjacency matrix A
        
        # random weights for the graph edges
        rand_wts = np.random.rand(n,3)
        alpha = rand_wts[:,0].flatten()
        C = binary_covariate_weights(nc, A, phi, sigma=0.1)
        #C = simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
        #C = covariate_weights_binary(C, minimal = 1/4, extreme = 4, phi=phi)
        
        # potential outcomes model
        fy = ppom(beta, C, alpha)

        # compute the true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        
        ####### Estimate ########

        # parameters for the staggered rollout - Cluster Randomized Design
        P = seq_treatment_probs(beta, target_p)        # treatment probabilities for each step of the staggered rollout on U
        P_prime = seq_treatment_probs(beta, p_prime)  # treatment probabilities for each step of the staggered rollout on the boundary of U
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

            #print("Design: {}, # of chosen clusters: {}".format(design, len(selected)))
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
                # Testing for edge case
                if isinstance(est, list):
                    with open('error_variables.picle', 'wb') as f:
                        pickle.dump({'A': A, 'P': P, 'H': H, 'selected': selected, 'U': selected_nodes, 'Z': Z, 'sums': sums, 'sums_U': sums_U, 'estimator': alg_names_clusterRD[x], 'estimate': est}, f)
                    raise ValueError('The estimator {} returned an invalid value {}'.format(alg_names_clusterRD[x], est))


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

def covariate_type(i, n, K, num=2):
    '''
    Returns the covariate type of unit i. Assumes that the number of communities is divisible by the number of covariate types.

    n (int) = number of units in the population
    K (int) = number of communities (blocks) in the SBM
    num (int)  = number of covariate types

    Example: Suppose n=8, K=4, and num = 2. 
        Communities 0 and 1 are assigned covariate type 0 and contain individuals 0,1,2,3
        Communities 2 and 3 are assigned covariate type 1 and contain individuals 4,5,6,7
        Individual i's covariate type is i // 4. Note that 4 = n // num.

    Example: Suppose n=8, K=2, and num = 2.
        Community 0 is assigned covariate type 0 and contains individuals 0,1,2,3
        Communities 1 is assigned covariate type 1 and contains individuals 4,5,6,7
        Individual i's covariate type is i // 4. Note that 4 = n // num.

    Example: Suppose n=8, K=4, and num = 4.
        Community 0 is assigned covariate type 0 and contains individuals 0,1
        Community 1 is assigned covariate type 1 and contains individuals 2,3
        Community 2 is assigned covariate type 2 and contains individuals 4,5
        Community 3 is assigned covariate type 3 and contains individuals 6,7
        Individual i's covariate type is i // 2. Note that 2 = n // num.
    '''
    assert num <= K and num%K==0, "there cannot be more covariate types than number of communities; number of types should divide evenly into the number of groups"
    div = n // num
    return i // div