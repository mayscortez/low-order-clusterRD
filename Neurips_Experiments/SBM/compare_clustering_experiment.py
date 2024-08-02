import sys
sys.path.insert(0, "../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle
import pymetis
import time
import itertools

startTime = time.time()
n = 1000
k = 20      # number of communities
c = n//k    # community size 
Cl_sbm = [list(range(c*i,c*(i+1))) for i in range(k)] # clustering based on SBM communities

Cl_anti = [list(range(i,n+i,k)) for i in range(k)] # anti-clustering

e_indeg = 5
e_outdeg = 4

pii = e_indeg/(c-1)
pij = e_outdeg/(n-c)

# parameters
betas = [2]             # model degrees
p = 0.1                   # treatment budget
qs = np.linspace(p,1,24)
r = 100                  # number of replications
g_reps = 10
stage1_designs = ['bernoulli','complete']
cluster_strategies = ["none","graph","community","random","anti"]

##############################################

data = {"beta":[], "q":[], "cl":[], "s1_design":[], "bias":[], "var":[], "var_s":[]}

def estimate_two_stage(fY,Cl,q,r,beta,cl_design):
    Q = np.linspace(0, q, beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    Z,U = staggered_rollout_two_stage(n,Cl,p,Q,r,design=cl_design) 
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p,Q))
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

# each of these is a dict of dicts of dicts of lists... 
# the outermost dictionary has keys corresponding to the model degrees (betas)
# the value corresponding to each beta is itself a dictionary with keys corresponding to clustering type ["none","graph","community","random","anti"]
# the value corresponding to each clustering type is a dictionary with keys corresponding to the cluster selection design ["bernoulli", "complete"]
# the value corresponding to each design is a dictionary with keys corresponding to the q values in qs
# the value corresponding to each q value is an empty list (to be filled later)
# e.g. Bias_dict[b][c][d][q] constains a list of the biases of the two-stage estimator under a model with degree b, a clustering c, and treatment probability q where clusters are chosen in the first stage with design d
TTE_hat_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}
Bias_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}
E_given_U_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}

for beta in betas:
    print("\nbeta={}".format(beta))
    for g in range(g_reps): # change this depending on how many graphs to average over
        if g%5==0:
            print("Graph iteration: {}".format(g))
        G = sbm(n,k,pii,pij)
        h = homophily_effects(G)

        clusterings = {"community":Cl_sbm, "anti":Cl_anti, "none":[]}

        # adjacency list representation
        A = [[] for _ in range(n)]
        for i,j in zip(*G.nonzero()):
            A[i].append(j)
            A[j].append(i)

        # create a clustering based on the graph
        _,membership = pymetis.part_graph(nparts=k,adjacency=A)
        membership = np.array(membership)
        Cl_graph = []
        for i in range(k):
            Cl_graph.append(np.where(membership == i)[0])

        clusterings["graph"] = Cl_graph
        #print(Cl_graph)

        # randomly chosen balanced clustering
        membership = np.array(list(range(k))*(c+1))[:n]
        np.random.shuffle(membership)

        Cl_random = []
        for i in range(k):
            Cl_random.append(np.where(membership == i)[0])

        clusterings["random"] = Cl_random

        fY = pom_ugander_yin(G,h,beta)
        TTE_true = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        # run the experiment (RCT + estimation)
        for cl, d in itertools.product(cluster_strategies,stage1_designs):
            if g%5==0:
                print("clustering: {}, design={}\n".format(cl,d))
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-2, verbose=5)(delayed(lambda q : estimate_two_stage(fY,clusterings[cl],q,r,beta,d))(q) for q in qs):
                #print(TTE_true, TTE_hat,'\n')
                Bias_dict[beta][cl][d][q].append(TTE_hat - TTE_true)
                TTE_hat_dict[beta][cl][d][q].append(TTE_hat)
                E_given_U_dict[beta][cl][d][q].append(E_given_U)

# save the data (?)
for beta in betas:
    for cl, d in itertools.product(cluster_strategies,stage1_designs):
        for q in qs:
            data["q"].append(q)
            data["cl"].append(cl)
            data["s1_design"].append(d)
            data["beta"].append(beta)
            data["bias"].append(np.average(Bias_dict[beta][cl][d][q]))
            data["var"].append(np.average((TTE_hat_dict[beta][cl][d][q] - np.average(TTE_hat_dict[beta][cl][d][q]))**2))
            data["var_s"].append(np.average((E_given_U_dict[beta][cl][d][q] - np.average(E_given_U_dict[beta][cl][d][q]))**2))

file = open("sbm_clusters_design.pkl", "wb") #sbm_clustering
pickle.dump((data), file)
file.close()

executionTime = (time.time() - startTime)
print('Total runtime in minutes: {}'.format(executionTime/60)) 
        

        