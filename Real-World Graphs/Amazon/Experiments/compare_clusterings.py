import sys
sys.path.insert(0, "../../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle
import pymetis

print("Loading Graph")

file = open("../Network/data.pkl", "rb")
G,Cls = pickle.load(file)
n = G.shape[0]

print("Calculating Homophily Effects")

h = homophily_effects(G)

# parameters
ncs = [100,300,500]         # number of clusters
p = 0.1                     # treatment budget
qs = np.linspace(p,1,16)    # effective treatment budget
r = 1000                    # number of replications

data = { "q": [], "clustering": [], "tte_hat": [], "est": [], "nc": [] }

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

fY = pom_market(G,h,2)
TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

for nc in ncs:
    print("Preparing Clusterings with {} Clusters".format(nc))

    clusterings = {}

    # cluster calculating using only feature data
    clusterings["feature"] = Cls[nc]

    # cluster calculated using graph data
    A = [[] for _ in range(n)]
    for i,j in zip(*G.nonzero()):
        A[i].append(j)

    _,membership = pymetis.part_graph(nparts=nc,adjacency=A)
    membership = np.array(membership)
    Cl_graph = []
    for i in range(nc):
        Cl_graph.append(np.where(membership == i)[0])

    clusterings["graph"] = Cl_graph

    # randomly chosen balanced clustering
    membership = np.array(list(range(nc))*(n//nc+1))[:n]
    np.random.shuffle(membership)

    Cl_random = []
    for i in range(nc):
        Cl_random.append(np.where(membership == i)[0])

    clusterings["random"] = Cl_random

    for label,Cl in clusterings.items():
        print("nc: {}\t Clustering: {}\t True TTE: {}".format(nc, label,TTE))

        for _ in range(r//1000):
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cl,q,1000,2))(q) for q in qs):
                data["q"] += [q]*2000
                data["clustering"] += [label]*2000
                data["nc"] += [nc]*2000
                data["est"] += ["real"]*1000
                data["tte_hat"] += list(TTE_hat - TTE)
                data["est"] += ["exp"]*1000
                data["tte_hat"] += list(E_given_U - TTE)

file = open("compare_clusterings_data.pkl", "wb")
pickle.dump((data), file)
file.close()