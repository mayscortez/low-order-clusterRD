from experiment_functions import *
from joblib import Parallel, delayed 
import matplotlib.pyplot as plt

### SBM Parameters
n = 210      # number of individuals
k = 10       # number of communities
ed = 10      # expected degree besides self
piis = list(np.arange(0.05,0.51,0.05))  # range from expected 1 neighbor in community to 10 neighbors in community

Cl = [list(range(c*(n//k),(c+1)*(n//k))) for c in range(k)]

alpha = np.random.rand(k)    # base effects in each community
f = lambda i, card_S, card_S_type_i: alpha[i] * 1.2**card_S_type_i / 3**card_S
#f = lambda i, card_S, card_S_type_i: np.random.rand()*10/(3**card_S)

### other parameters
beta = 2                  # model degree
p = 0.2                   # treatment budget 
qs = np.arange(p,1,0.05)  # effective treatment budget in selected clusters
r = 1000                  # number of replications per graph
R = 10                    # number of graphs

def simulate_pii(pii):
    pij = (ed-((n//k-1)*pii+1))/(n-n//k)

    MSEs = []
    for q in qs:
        Q = (np.arange(10)*q/beta)[:beta+1]
        MSE = 0

        for _ in range(R):
            G = sbm(n,k,pii,pij)
            C = community_based_weights(G,Cl,f,beta)
            Y = outcomes(G,C,beta)
            TTE = np.sum(Y(np.ones(n))-Y(np.zeros(n)))/n

            Z = staggered_rollout_bern_cluster(n,Cl,p/q,Q,r)

            TTE_ests_s = pi_estimate_tte_clustered(Z,Y,p/q,Q)
            MSE += sum((TTE_ests_s-TTE)**2)/r
        
        MSEs.append(MSE/R)
    
    return (pii,MSEs)

for (pii,MSEs) in Parallel(n_jobs=-1, verbose=10)(delayed(simulate_pii)(pii) for pii in piis):
    plt.plot(qs,MSEs,label=pii)
plt.xlabel("q")
plt.ylabel("MSE")
plt.legend()
plt.show()