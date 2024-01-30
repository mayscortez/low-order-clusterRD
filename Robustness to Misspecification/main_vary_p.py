'''
Compare the MSE of the estimators under cluster staggered rollout versus bernoulli staggerred rollout with misspecified model
'''

# Setup
import numpy as np
import pandas as pd
import time
from experiment_functions import PI, horvitz_thompson, DM_naive, DM_fraction, poly_LS_prop, poly_LS_num, SBM, binary_covariate_weights, ppom, threshold_model, saturation_model, seq_treatment_probs, bern_coeffs, select_clusters_complete, select_clusters_bernoulli, staggered_rollout_bern_clusters, outcome_sums, staggered_rollout_bern


def main(model, graphNum, T, U, p=0.06, phi=0, p_in = 0.5, cluster_selection = "bernoulli"):
    '''
    model (dict) 
        key: 'type', value: a string denoting with model to use (e.g. 'ppom' for polynomial)
        key: 'degree', value: an int denoting the degree of the model (e.g. beta if model == 'ppom')
        key: 'name', value: a string denoting the full model name (e.g. 'ppom3' for polynomial of degree 3)
        key: 'params', value: a list of additional model parameters if there are any
    graphNum (int): how many graph models to average over
    U (int): how many cluster samples to average over
    T (int): how many treatment samples to average over
    p (float): treatment budget (marginal treatment probability)
    phi (float): probability that an individual switches covariate type
    p_in (float): edge probability within a cluster
    cluster_selection_RD (str): either "complete" or "bernoulli" depending on the design used for selecting clusters
    '''
    experiment = 'vary_p' 
    degree = model['degree']
    model_name = model['name']
    if model["type"] == 'ppom':
        save_path = 'output/' + experiment + '/'  + model_name + '/' + cluster_selection + '/'    
    else:
        save_path = 'output/' + experiment + '/'  + model_name + '-ppom' + str(degree) + '/' + cluster_selection + '/'  
    
    p_out = (0.5-p_in)/49
    if p_in != p_out:
        graphStr = "SBM"    # stochastic block model    
    else:
        graphStr = "ER"     # erdos-renyi
    
    n = 1000            # number of nodes
    nc = 50             # number of communities
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_p' + str(p).replace('.','') + '_phi' + str(phi).replace('.','') # naming convention
    
    f = open(save_path + experiment + '-' + model_name + fixed + '_' +  cluster_selection + '.txt', 'w') # e.g filename could be vary_phi-ppom1_n1000_nc50_in005_out0_B01_p1_bernoulli.txt 

    ######################################################################
    # Run Experiment: Increasing the treatment probability within clusters
    ######################################################################
    startTime1 = time.time()
    results = []
    probs = [0.06, 0.12, 0.25, 1/3, 2/3, 1]

    for q in probs:
        print("treatment prob q = {}".format(q))
        startTime2 = time.time()
        K_expected = int(p * nc / q)

        results.extend(run_experiment(model, n, nc, p, q, p_in, p_out, phi, cluster_selection, K_expected, graphNum, T, U, graphStr))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for q = {} step: {}'.format(q, executionTime),file=f)
        print('Runtime (in seconds) for q = {} step: {}'.format(q, executionTime))

    executionTime = (time.time() - startTime1)
    print('\nTotal runtime in minutes: {}'.format(executionTime/60),file=f)   
    print('\nTotal runtime in minutes: {}'.format(executionTime/60))      
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + experiment + '-' + model_name + fixed + '_' +  cluster_selection + '-full.csv')
    

def run_experiment(model, n, nc, p, q, Pii, Pij, phi, design, EK, graphNum, T, U, graphStr, betas = [1,2,3], p_prime=0, realized = True):
    '''
    model (dict) 
        key: 'type', value: a string denoting with model to use (e.g. 'ppom' for polynomial)
        key: 'degree', value: true degree of the model (e.g. [beta] if model == 'ppom')
        key: 'name', value: a string denoting the full model name (e.g. 'ppom3')
        key: 'params', value: a list of additional model parameters if there are any
    n = population size
    nc = number of clusters
    p = original treatment budget/fraction
    q = treatment probability for chosen clusters
    Pii = edge probability within communities
    Pij = edge prob btwn different communities
    phi = correlation btwn community & effect type (probability between 0 and 0.5)
    design = design being used for selecting clusters, either "complete" or "bernoulli"
    EK = expected value of K E[K] (i.e. expected number of chosen clusters)
    graphNum = number of graphs to average over
    T = number of trials per graph
    graphStr = type of graph e.g. "SBM" for stochastic block model or "ER" for Erdos-Renyi
    p_prime = the budget on the boundary of U
    '''

    dict_base = {'n': n, 'nc': nc, 'Pii': Pii, 'Pij': np.round(Pij,3), 'Phi': phi, 'p': p, 'q': q, 'EK': EK}
    
    # Cluster Randomized Design Estimators
    estimatorsClRD = []

    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: PI(n*x,sums1,H_m1))        # estimator looks at all [n], beta=1
    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: PI(n*x,sums_U1,H_m1))      # estimator only looking at [U], beta=1

    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: PI(n*x,sums2,H_m2))         # estimator looks at all [n], beta=2
    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: PI(n*x,sums_U2,H_m2))       # estimator only looking at [U], beta=2

    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: PI(n*x,sums3,H_m3))         # estimator looks at all [n], beta=3
    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: PI(n*x,sums_U3,H_m3))       # estimator only looking at [U], beta=3

    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: horvitz_thompson(n, nc, y1, A, z1, EK/nc, p))  

    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: DM_naive(y1,z1))  
    estimatorsClRD.append(lambda x,y1,z1,sums1,H_m1,sums_U1,y2,z2,sums2,H_m2,sums_U2,y3,z3,sums3,H_m3,sums_U3: DM_fraction(n,y1,A,z1,0.75))  


    names_ClRD = ['PI$(q;1)$', 'PI-$\mathcal{U}(q;1)$', 
                  'PI$(q;2)$', 'PI-$\mathcal{U}(q;2)$',
                    'PI$(q;3)$', 'PI-$\mathcal{U}(q;3)$',
                    'HT', 'DM-C', 'DM-C($0.75$)']

    # Bernoulli Randomized Design Estimators
    estimatorsBRD = []
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: PI(n,sums1,H_m1))              # beta=1
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: poly_LS_prop(1, y1,A,z1))      # polynomial LS, beta=1
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: poly_LS_num(1, y1,A,z1))

    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: PI(n,sums2,H_m2))              # beta=2 
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: poly_LS_prop(2, y2,A,z2))      # polynomial LS, beta=1
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: poly_LS_num(2, y2,A,z2))

    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: PI(n,sums3,H_m3))              # beta=3
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: poly_LS_prop(3, y3,A,z3))      # polynomial LS, beta=1
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: poly_LS_num(3, y3,A,z3))

    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: DM_naive(y1,z1))                 # difference in means 
    estimatorsBRD.append(lambda y1,z1,sums1,H_m1,y2,z2,sums2,H_m2,y3,z3,sums3,H_m3: DM_fraction(n,y1,A,z1,0.75))     # thresholded difference in means

    names_BRD = ['PI$(p;1)$', 'LS-Prop(1)', 'LS-Num(1)',
                'PI$(p;2)$', 'LS-Prop(2)', 'LS-Num(2)',
                'PI$(p;3)$', 'LS-Prop(3)', 'LS-Num(3)','DM', 'DM($0.75$)']

    results = []
    for g in range(graphNum):
        graph_rep = str(g)
        dict_base.update({'Graph':graphStr+graph_rep})
        
        G,A = SBM(n, nc, Pii, Pij)  #returns the SBM networkx graph object G and the corresponding adjacency matrix A  

        # random weights for the graph edges
        rand_wts = np.random.rand(n,3)
        alpha = rand_wts[:,0].flatten()
        C = binary_covariate_weights(nc, A, phi, sigma=0.1)
        
        # potential outcomes model
        if model['type'] == 'ppom':
            beta = model['degree']
            fy = ppom(beta, C, alpha)
        if model['type'] == 'threshold':
            beta = model['degree']
            theta = model['params'][0]
            type = model['params'][1]
            fy = threshold_model(theta, beta, C, alpha, A, type)
        if model['type'] == 'saturation':
            beta = model['degree']
            theta = model['params'][0]
            fy = saturation_model(theta, beta, C, alpha)
            

        # compute the true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))  
        dict_base.update({'TTE': np.round(TTE,5)})      

        ####### Estimate ########
        P = []
        P_prime = []
        H = []
        P_bern = []
        H_bern = []
        for b in betas:
            # parameters for the staggered rollout - Cluster Randomized Design
            P.append(seq_treatment_probs(b, q))        # treatment probabilities for each step of the staggered rollout on U
            P_prime.append(seq_treatment_probs(b, p_prime))  # treatment probabilities for each step of the staggered rollout on the boundary of U
            H.append(bern_coeffs(P[b-1]))                      # coefficients for the polynomial interpolation estimator

            # parameters for the staggered rollout - Bernoulli Randomized Design
            P_bern.append(seq_treatment_probs(b, p)) # treatment probabilities for each step of the staggered rollout on [n]
            H_bern.append(bern_coeffs(P_bern[b-1]))          # coefficients for the polynomial interpolation estimator    

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

            for j in range(T):
                dict_base.update({'rep_z': j})
                # Cluster Randomized Design
                Z1 = staggered_rollout_bern_clusters(n, selected_nodes, P[0], [], P_prime[0])
                Z2 = staggered_rollout_bern_clusters(n, selected_nodes, P[1], [], P_prime[1])
                Z3 = staggered_rollout_bern_clusters(n, selected_nodes, P[2], [], P_prime[2])
                z1 = Z1[1,:]
                z2 = Z2[2,:]
                z3 = Z3[3,:]
                y1 = fy(z1)
                y2 = fy(z2)
                y3 = fy(z3)
                sums1, sums_U1 = outcome_sums(n, fy, Z1, selected_nodes) # the sums corresponding to all nodes (i.e. [n]) and just selected nodes (i.e. [U])
                sums2, sums_U2 = outcome_sums(n, fy, Z2, selected_nodes)
                sums3, sums_U3 = outcome_sums(n, fy, Z3, selected_nodes)

                for x in range(len(estimatorsClRD)):
                    if realized:
                        bias_correction = 1 - (1-(EK/nc))**nc
                        est = estimatorsClRD[x]((K_real/nc)*bias_correction,y1,z1,sums1,H[0],sums_U1,y2,z2,sums2,H[1],sums_U2,y3,z3,sums3,H[2],sums_U3)
                    else:
                        est = estimatorsClRD[x](EK/nc,y1,z1,sums1,H[0],sums_U1,y2,z2,sums2,H[1],sums_U2,y3,z3,sums3,H[2],sums_U3)                
                    dict_base.update({'Estimator': names_ClRD[x], 'est': np.round(est,5), 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Rel_bias_sq':((est-TTE)/TTE)**2, 'Bias_sq': ((est-TTE)**2)})
                    results.append(dict_base.copy())

            # Bernoulli Randomized Design (No Clusters)
            dict_base.update({'design': 'Bernoulli'})
            Z1_bern = staggered_rollout_bern(n, P_bern[0])
            Z2_bern = staggered_rollout_bern(n, P_bern[1])
            Z3_bern = staggered_rollout_bern(n, P_bern[2])
            z1_bern = Z1_bern[1,:]
            z2_bern = Z2_bern[2,:]
            z3_bern = Z3_bern[3,:]
            y1_bern = fy(z1_bern)
            y2_bern = fy(z2_bern)
            y3_bern = fy(z3_bern)
            sums1_bern = outcome_sums(n, fy, Z1_bern, range(0,n))[0]
            sums2_bern = outcome_sums(n, fy, Z2_bern, range(0,n))[0]
            sums3_bern = outcome_sums(n, fy, Z3_bern, range(0,n))[0]

            for x in range(len(estimatorsBRD)):
                est = estimatorsBRD[x](y1_bern,z1_bern,sums1_bern,H_bern[0],y2_bern,z2_bern,sums2_bern,H_bern[1],y3_bern,z3_bern,sums3_bern,H_bern[2]) # have it include both the parameters for all as well as just U
                dict_base.update({'Estimator': names_BRD[x], 'est':est, 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Rel_bias_sq':((est-TTE)/TTE)**2, 'Bias_sq': ((est-TTE)**2)})
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