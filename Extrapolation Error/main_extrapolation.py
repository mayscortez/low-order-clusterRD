'''
How does the bias and variance of the estimator change as we reduce the extrapolation error (increase p)?
'''

# Setup
import numpy as np
import pandas as pd
import time
from experiment_functions import PI, horvitz_thompson, DM_naive, DM_fraction, poly_LS_prop, poly_LS_num, SBM, binary_covariate_weights, ppom, seq_treatment_probs, bern_coeffs, select_clusters_complete, select_clusters_bernoulli, staggered_rollout_bern_clusters, outcome_sums, staggered_rollout_bern
import pickle

def main(beta, graphNum, T, U, p=0.06, phi=0, p_in=0.5, cluster_selection_RD = "bernoulli"):
    '''
    beta (int): degree of the model
    graphNum (int): how many graph models to average over
    U (int): how many cluster samples to average over
    T (int): how many treatment samples to average over
    p (float): treatment budget (marginal treatment probability)
    cluster_selection_RD (str): either "complete" or "bernoulli" depending on the design used for selecting clusters
    '''
    deg_str = '_deg' + str(beta)  # naming convention
    save_path = 'output/' + 'deg' + str(beta) + '/' + cluster_selection_RD + '/' # e.g. save to the folder output/deg1/bernoulli/
    experiment = 'extrapolation'                    
    
    graphStr = "SBM"          # stochastic block model    
    n = 1000                  # number of nodes
    nc = 50                   # number of communities
    p_out = (0.5-p_in)/49     # edge probability between different communities, this way keeps expected degree constant at 10
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_p' + str(p).replace('.','') + '_phi' + str(np.round(phi,3)).replace('.','') # naming convention
    
    f = open(save_path + experiment + fixed + deg_str + '_' + cluster_selection_RD + '.txt', 'w') # e.g filename could be correlation_n1000_nc50_in005_out0_p1_B01_deg2_bernoulli.txt 
    startTime1 = time.time()

    #################################################################
    # Run Experiment: Increasing the correlation between covariates
    #################################################################
    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    
    results = []
    if p == 0.06:
        probs = [3/50, 3/25, 3/18, 3/12, 3/9, 3/7, 3/5, 3/4, 1]  # K in [50, 25, 18, 12, 9, 7, 5, 4, 3]
    elif p == 0.02:
        probs = [1/50, 1/25, 1/18, 1/12, 1/8, 1/4, 1/3, 1/2, 1]  # K in [50, 25, 18, 12, 8, 4, 3, 2, 1] 
    else:
        raise ValueError("Need to set probs in main_extrapolation for this value of p")

    for q in probs:
        print("treatment prob q = {}".format(q))
        startTime2 = time.time()

        K_expected = int(p * nc / q)   # if complete design, this is actual number of chosen clusters; if Bernoulli this is expected number of chosen clusters

        results.extend(run_experiment(beta, n, nc, p, q, r, diag, p_in, p_out, phi, cluster_selection_RD, K_expected, graphNum, T, U, graphStr))

        executionTime = (time.time() - startTime2)
        print('Runtime (in minutes) for q = {}, E[K] = {} step: {}'.format(np.round(q,3), K_expected, executionTime/60),file=f)
        print('Runtime (in minutes) for q = {}, E[K] = {} step: {}'.format(np.round(q,3), K_expected, executionTime/60))

    executionTime = (time.time() - startTime1)
    print('Degree: {}'.format(beta)) 
    print('Total runtime in hours: {}'.format(executionTime/3600),file=f)   
    print('Total runtime in hours: {}'.format(executionTime/3600)) 
    print('')       
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + experiment + fixed + deg_str + '_' + cluster_selection_RD + '-full.csv') # e.g filename could be correlation_n1000_nc50_in005_out0_p1_B01_deg2_bernoulli-full.csv 
    

def run_experiment(beta, n, nc, p, q, r, diag, Pii, Pij, phi, design, EK, graphNum, T, U, graphStr, p_prime=0, realized = True):
    '''
    beta = degree of the model / polynomial
    n = population size
    nc = number of clusters
    p = original treatment budget/fraction
    q = the treatment probability within clusters i.e. the target treatment probability - conditional treatment prob
    r = ratio offdiag/diag: (indirect effect)/(direct effects)
    diag = maxium norm of the direct effects before covariate type scaling
    Pii = edge probability within communities
    Pij = edge prob btwn different communities
    phi = correlation btwn community & effect type (probability between 0 and 0.5)
    design = design being used for selecting clusters, either "complete" or "bernoulli"
    EK = expected value of K E[K] (i.e. expected number of chosen clusters)
    graphNum = number of graphs to average over
    T = number of trials per graph
    graphStr = type of graph e.g. "SBM" for stochastic block model or "ER" for Erdos-Renyi
    p_prime = the budget on the boundary of U
    realized = if True, uses realized value of K/nc in the estimator instead of expected (this only changes things if we use Bernoulli design to choose clusters)
    '''
    
    offdiag = r*diag   # maximum norm of indirect effect

    dict_base = {'n': n, 'nc': nc, 'Pii': Pii, 'Pij': np.round(Pij,3), 'Phi': phi, 'q': np.round(q,3), 'p': p, 'EK': EK, 'ratio': r}

    # Cluster Randomized Design Estimators
    estimators_clusterRD = []
    estimators_clusterRD.append(lambda x,y,z,sums,H_m,sums_U: PI(n*x,sums,H_m))             # estimator looks at all [n]
    estimators_clusterRD.append(lambda x,y,z,sums,H_m,sums_U: PI(n*x,sums_U,H_m))           # estimator only looking at [U]
    estimators_clusterRD.append(lambda x,y,z,sums,H_m,sums_U: horvitz_thompson(n, nc, y, A, z, EK/nc, q))  
    estimators_clusterRD.append(lambda x,y,z,sums,H_m,sums_U: DM_naive(y,z))                 # difference in means 
    estimators_clusterRD.append(lambda x,y,z,sums,H_m,sums_U: DM_fraction(n,y,A,z,0.75))     # thresholded difference in means

    alg_names_clusterRD = ['PI($q$)', 'PI-$\mathcal{U}$($q$)', 'HT', 'DM-C', 'DM-C($0.75$)']

    # Bernoulli Randomized Design Estimators
    estimators_bernRD = []
    estimators_bernRD.append(lambda y,z,sums,H_m: PI(n,sums,H_m))
    estimators_bernRD.append(lambda y,z, sums, H_m: poly_LS_prop(beta, y,A,z))      # polynomial regression
    estimators_bernRD.append(lambda y,z, sums, H_m: poly_LS_num(beta, y,A,z))
    estimators_bernRD.append(lambda y,z,sums,H_m: DM_naive(y,z))                 # difference in means 
    estimators_bernRD.append(lambda y,z,sums,H_m: DM_fraction(n,y,A,z,0.75))     # thresholded difference in means

    alg_names_bernRD = ['PI($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']

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
        dict_base.update({'TTE': np.round(TTE,5)})
        
        ####### Estimate ########

        # parameters for the staggered rollout - Cluster Randomized Design
        P = seq_treatment_probs(beta, q)        # treatment probabilities for each step of the staggered rollout on U
        P_prime = seq_treatment_probs(beta, p_prime)  # treatment probabilities for each step of the staggered rollout on the boundary of U
        H = bern_coeffs(P)                      # coefficients for the polynomial interpolation estimator

        # parameters for the staggered rollout - Bernoulli Randomized Design
        P_bernRD = seq_treatment_probs(beta, p) # treatment probabilities for each step of the staggered rollout on [n]
        H_bern = bern_coeffs(P_bernRD)          # coefficients for the polynomial interpolation estimator

        for i in range(U):
            # select clusters 
            if design == "complete":
                selected = select_clusters_complete(nc, EK)
                K_real = EK
            else:
                selected = select_clusters_bernoulli(nc, EK/nc)
                K_real = len(selected)

            selected_nodes = [x for x,y in G.nodes(data=True) if (y['block'] in selected)] # get the nodes in selected clusters

            dict_base.update({'rep_U': i, 'design': 'Cluster', 'K': K_real})

            EZ = np.zeros(shape=(beta+1,n))
            for b in range(1,beta+1):
                EZ[b,selected_nodes] = P[b]

            Esums = outcome_sums(n, fy, EZ, selected_nodes)[0]

            bias_correction = 1 - (1-(EK/nc))**nc
            conditional_E = PI(n*(K_real/nc)*bias_correction, Esums, H)

            dict_base.update({'Estimator': 'E[PI$(q)|\mathcal{U}$]', 'est': np.round(conditional_E,5), 'Bias': (conditional_E-TTE)/TTE, 'Abs_Bias': (conditional_E-TTE), 'Rel_bias_sq':((conditional_E-TTE)/TTE)**2, 'Bias_sq': ((conditional_E-TTE)**2)})
            results.append(dict_base.copy())

            for j in range(T):
                dict_base.update({'rep_z': j})
                # Cluster Randomized Design
                Z = staggered_rollout_bern_clusters(n, selected_nodes, P, [], P_prime)
                z = Z[beta,:]
                y = fy(z)
                sums, sums_U = outcome_sums(n, fy, Z, selected_nodes) # the sums corresponding to all nodes (i.e. [n]) and just selected nodes (i.e. [U])

                for x in range(len(estimators_clusterRD)):
                    if realized:
                        bias_correction = 1 - (1-(EK/nc))**nc
                        est = estimators_clusterRD[x]((K_real/nc)*bias_correction,y,z,sums,H,sums_U)
                    else:
                        est = estimators_clusterRD[x](EK/nc,y,z,sums,H,sums_U)
                    dict_base.update({'Estimator': alg_names_clusterRD[x], 'est': np.round(est,5), 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Rel_bias_sq':((est-TTE)/TTE)**2, 'Bias_sq': ((est-TTE)**2)})
                    results.append(dict_base.copy())

            # Bernoulli Randomized Design (No Clusters)
            dict_base.update({'design': 'Bernoulli'})
            Z_bern = staggered_rollout_bern(n, P_bernRD)
            z_bern = Z_bern[beta,:]
            y_bern = fy(z)
            sums_bern = outcome_sums(n, fy, Z_bern, range(0,n))[0]

            for x in range(len(estimators_bernRD)):
                est = estimators_bernRD[x](y_bern,z_bern,sums_bern,H_bern) # have it include both the parameters for all as well as just U
                dict_base.update({'Estimator': alg_names_bernRD[x], 'est': np.round(est,5), 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Rel_bias_sq':((est-TTE)/TTE)**2, 'Bias_sq': ((est-TTE)**2)})
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
