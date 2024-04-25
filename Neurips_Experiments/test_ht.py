import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from experiment_functions import *
from joblib import Parallel, delayed 

nf = open("Amazon/Network/data.pkl",'rb')
G,Cls = pickle.load(nf)
nf.close()

n = G.shape[0]
h = homophily_effects(G)

beta = 2
fY = pom_ugander_yin(G,h,beta)
TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
print(f"Actual: {TTE}")

p = 0.1
q = 1

nc = 500
Cl = Cls[nc]

bias = []
sd = []

def estimate_tte(q):
    Z,_ = staggered_rollout_two_stage(n,Cl,p,[q],1000) 
    Y = fY(Z)
    return ht_estimate_tte(Z[0,:,:],Y[0,:,:],G,Cl,p,q)

qs = np.linspace(p,1,4)
#for TTE_hat in Parallel(n_jobs=-1)(delayed(estimate_tte)(q) for q in qs):
for q in qs:
    print(f"q={q}")
    TTE_hat = estimate_tte(q)
    bias.append(np.average(TTE_hat - TTE))
    sd.append(np.average((TTE_hat - np.average(TTE_hat))**2)**(1/2))
bias = np.array(bias)
sd = np.array(sd)

sns.set_theme()

plt.plot(qs,bias,color="tab:blue")
plt.fill_between(qs, bias-sd, bias+sd, color="tab:blue",alpha=0.2)
plt.show()

