import sys
sys.path.insert(0, "../../")

from experiment_functions import *
import pickle

from kmodes.kmodes import KModes
from joblib import Parallel, delayed 

print("Loading Graph")

graph_file = open("../Network/data.pkl", "rb")
G,features = pickle.load(graph_file)
n = G.shape[0]

print("Calculating Homophily Effects")

h = homophily_effects(G)

# parameters
betas = [1,2]               # model degree
ncs = [10,20,30,40]         # number of clusters
p = 0.1                     # treatment budget
qs = np.linspace(p,1,19)    # effective treatment budget
r = 10000                   # number of replications

##############################################

data = { "q": [], "nc": [], "beta": [], "tte_hat": [], "est": [] }

def estimate_two_stage(fY,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for nc in ncs:
    print("Assigning Units to {} Clusters".format(nc))

    membership = KModes(n_clusters=nc, init="random", n_init=100, n_jobs=-1).fit_predict(features)

    Cl = []
    for i in range(nc):
        Cl.append(list(np.where(membership == i)[0]))

    print("Cluster Sizes:",[len(cl) for cl in Cl])

    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("beta: {}\t True TTE: {}".format(beta,TTE))
        
        for _ in range(r//1000):
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,q,1000,beta))(q) for q in qs):
                data["q"] += [q]*2000
                data["beta"] += [beta]*2000
                data["nc"] += [nc]*2000
                data["est"] += ["real"]*1000
                data["tte_hat"] += list(TTE_hat - TTE)
                data["est"] += ["exp"]*1000
                data["tte_hat"] += list(E_given_U - TTE)

file = open("pi_data.pkl", "wb")
pickle.dump((data), file)
file.close()