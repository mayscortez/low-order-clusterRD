'''
TODO: description
'''

# Setup
import numpy as np
import pandas as pd
import sys
import time
from myFunctions import *

beta = 2
deg_str = '_deg' + str(beta)
save_path = 'outputFiles/' + 'degree' + str(beta) + '/'

def main():
    G = 20
    T = 20               # number of trials per cluster size
    graphStr = "120lat"   # square lattice

    
    f = open(save_path +'experiments_output' + deg_str + '.txt', 'w')

    startTime1 = time.time()
    ###########################################
    # Run Experiment: Varying Cluster Size
    ###########################################
    N = 120*120
    n = 120

    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    
    results = []
    sizes = np.array([1,2,3,4,5,6,8,10,12,15,20,24,30,50,60,120])
    #sizes = [1,10,20]
    B = 0.05        # treatment probability
    q = 0.9

    for k in sizes:
        print("k = {}".format(k))
        startTime2 = time.time()

        results.extend(run_experiment(G,T,N,B,r,k,graphStr,diag=diag,beta=beta,q=q))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for k = {} step: {}'.format(k,executionTime),file=f)
        print('Runtime (in seconds) for k = {} step: {}'.format(k,executionTime))

    executionTime = (time.time() - startTime1)
    print('Runtime (size experiment) in minutes: {}'.format(executionTime/60),file=f)   
    print('Runtime (size experiment) in minutes: {}'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + graphStr + '-' + str(q).replace('.','') + '_'+ 'clusterSize-full-data'+deg_str +'.csv')
    
    
    ###########################################
    # Run Experiment: Varying q
    ###########################################
    N = 120*120
    n = 120
    k = 12           # cluster size k by k
    B = 0.05         # treatment budget

    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    
    results = []
    cluster_frac = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #cluster_frac = [0.2, 0.3]

    for q in cluster_frac:
        print("q = {}".format(q))
        startTime3 = time.time()

        results.extend(run_experiment(G,T,N,B,r,k,graphStr,diag=diag,beta=beta,q=q))

        executionTime = (time.time() - startTime3)
        #print('Runtime (in seconds) for q = {} step: {}'.format(q,executionTime),file=f)
        print('Runtime (in seconds) for q = {} step: {}'.format(q,executionTime))

    executionTime = (time.time() - startTime2)
    print('Runtime (cluster fraction experiment) in minutes: {}'.format(executionTime/60),file=f)   
    print('Runtime (cluster fraction experiment) in minutes: {}'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + graphStr + '-' + str(k) + '_' + 'clusterFraction-full-data'+ deg_str +'.csv')

    
    ###########################################
    # Run Experiment: Varying p
    ###########################################
    N = 120*120
    n = 120
    k = 12           # cluster size k by k
    B = 0.05         # treatment budget

    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    
    results = []
    new_budget = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #new_budget = [0.2, 0.5]

    for p in new_budget:
        print("p = {}".format(p))
        startTime4 = time.time()

        results.extend(run_experiment(G,T,N,B,r,k,graphStr,diag=diag,beta=beta,p=p))

        executionTime = (time.time() - startTime4)
        print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime),file=f)
        print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime))

    executionTime = (time.time() - startTime3)
    print('Runtime (new budget experiment) in minutes: {}'.format(executionTime/60),file=f)   
    print('Runtime (new budget experiment) in minutes: {}'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path + graphStr + '-' + str(k) + '_' + 'newBudget-full-data'+ deg_str+'.csv')

    executionTime = (time.time() - startTime1)
    print('Runtime (whole script) in minutes: {}'.format(executionTime/60),file=f)
    print('Runtime (whole script) in minutes: {}'.format(executionTime/60))
    
    sys.stdout.close()
    

def run_experiment(G,T,N,B,r,k,graphStr,diag=1,beta=1,q=-1,p=-1):
    '''
    G = number of graphs to average over
    T = number of trials per graph
    N = total population size (N=n*n)
    B = original treatment budget/fraction
    r = ratio offdiag/diag (indirect effect)/(direct effects)
    k = cluster size length i.e. clusters are k by k
    '''
    
    offdiag = r*diag   # maximum norm of indirect effect

    results = []
    dict_base = {'B': B, 'ratio': r, 'N': N, 'k': k}

    #sz = str(N) + '-'
    n = int(np.sqrt(N))
    A = lattice2Dsq(n,n)
    for g in range(G):
        graph_rep = str(g)
        dict_base.update({'Graph':graphStr+graph_rep})

        rand_wts = np.random.rand(N,3)
        alpha = rand_wts[:,0].flatten()
        C = simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
        
        # potential outcomes model
        fy = ppom(beta, C, alpha)

        # compute and print true TTE
        TTE = 1/N * np.sum((fy(np.ones(N)) - fy(np.zeros(N))))
        #print("The true TTE is {}".format(TTE))

        NC = int(np.ceil(n/k)**2) # number of clusters 
        clusters_flat = bf_clusters(NC,N)[1]
        if q == -1:
            q = B / p
        K = int(np.ceil(q*NC))
        q_true = K/NC
        

        if p == -1:
            p = NC * B / K
        dict_base.update({'p': p, 'NC': NC, 'q': q, 'q_true':q_true})
        
        P = seq_treatment_probs(beta, p)
        H = bern_coeffs(P)

        ####### Estimate ########
        estimators = []
        estimators.append(lambda y,z,sums,H_m: graph_agnostic(N,sums,H_m)/q_true)
        estimators.append(lambda y,z, sums, H_m: poly_regression_prop(beta, y,A,z))
        estimators.append(lambda y,z, sums, H_m: poly_regression_num(beta, y,A,z))
        estimators.append(lambda y,z,sums,H_m: diff_in_means_naive(y,z))
        estimators.append(lambda y,z,sums,H_m: diff_in_means_fraction(N,y,A,z,0.75))


        #alg_names = ['PI($p$)', 'DM', 'DM($0.75$)']
        alg_names = ['PI($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']


        bern_est = [0,1,2,3,4]

        for i in range(T):
            selected = select_clusters_complete(NC, K)
            selected_nodes = idx_of_U(selected, clusters_flat)

            dict_base.update({'rep': i})
            Z = staggered_rollout_bern(N, P, selected_nodes)
            z = Z[beta,:]
            y = fy(z)
            sums = outcome_sums(fy, Z)

            for x in bern_est:
                est = estimators[x](y,z,sums,H)
                dict_base.update({'Estimator': alg_names[x], 'Bias': (est-TTE)/TTE, 'Abs_Bias': (est-TTE), 'Bias_sq': ((est-TTE)**2)})
                results.append(dict_base.copy())

    return results


if __name__ == "__main__":
    main()