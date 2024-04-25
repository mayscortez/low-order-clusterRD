import sys
sys.path.insert(0, "../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle
import pymetis

n = 800
k = 20
c = n//k    # community size 
Cl_sbm = [list(range(c*i,c*(i+1))) for i in range(k)]

Cl_anti = [list(range(i,n+i,k)) for i in range(k)]

e_indeg = 5
e_outdeg = 4

pii = e_indeg/(c-1)
pij = e_outdeg/(n-c)

# parameters
beta = 2                  # model degree
p = 0.1                   # treatment budget
qs = np.linspace(p,1,24)
r = 1000                  # number of replications

##############################################

data = { "q":[], "clustering":[], "bias":[], "var":[], "var_s":[] }

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    Z,U = staggered_rollout_two_stage(n,Cl,p,Q,r) 
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p,Q))
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)


TTE_hat_dict = {c: {q:[] for q in qs} for c in ["graph","community","random","anti"]}
Bias_dict = {c: {q:[] for q in qs} for c in ["graph","community","random","anti"]}
E_given_U_dict = {c: {q:[] for q in qs} for c in ["graph","community","random","anti"]}

for _ in range(10):
    G = sbm(n,k,pii,pij)
    h = homophily_effects(G)

    clusterings = {"community":Cl_sbm, "anti":Cl_anti}

    # adjacency list representation
    A = [[] for _ in range(n)]
    for i,j in zip(*G.nonzero()):
        A[i].append(j)
        A[j].append(i)

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

    for cl in ["graph","community","random","anti"]:
        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=5)(delayed(lambda q : estimate_two_stage(fY,clusterings[cl],q,r,beta))(q) for q in qs):
            Bias_dict[cl][q].append(TTE_hat - TTE_true)
            TTE_hat_dict[cl][q].append(TTE_hat)
            E_given_U_dict[cl][q].append(E_given_U)

for cl in ["graph","community","random","anti"]:
    for q in qs:
        data["q"].append(q)
        data["clustering"].append(cl)
        data["bias"].append(np.average(Bias_dict[cl][q]))
        data["var"].append(np.average((TTE_hat_dict[cl][q] - np.average(TTE_hat_dict[cl][q]))**2))
        data["var_s"].append(np.average((E_given_U_dict[cl][q] - np.average(E_given_U_dict[cl][q]))**2))

file = open("sbm_clustering.pkl", "wb")
pickle.dump((data), file)
file.close()
        

        