import pickle

from experiment_functions import *
from joblib import Parallel, delayed 

def estimate_two_stage(fY,n,p,q,r,beta):
    Q = np.linspace(0,q,beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    for _ in range(r//1000):
        Z,U = staggered_rollout_two_stage_unit(n,p,Q,1000) 
        tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p,Q))
        e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))
        
    return (q, tte_hat, e_tte_hat_given_u)

# def estimate_two_stage_random(fY,n,p,q,r,beta):
#     Q = np.linspace(0,q,beta+1)

#     tte_hat = []
#     e_tte_hat_given_u = []
#     for _ in range(r//10):
#         nc = 20
#         membership = np.array(list(range(nc))*(n//nc+1))[:n]
#         np.random.shuffle(membership)

#         Cl = []
#         for i in range(nc):
#             Cl.append(np.where(membership == i)[0])

#         Z,U = staggered_rollout_two_stage(n,Cl,p,Q,10) 
#         tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p,Q))
#         e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

#     return (q, tte_hat, e_tte_hat_given_u)

def estimate_two_stage_restricted(fY,n,p,q,r,beta):
    Q = np.linspace(0,q,beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    for _ in range(r//1000):
        Z,U = staggered_rollout_two_stage_unit(n,p,Q,1000) 
        tte_hat = np.append(tte_hat,two_stage_restricted_estimator(fY(Z),U,p,Q))
        e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(U*(fY(U) - fY(np.zeros(n))),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

infile = open("Email/Network/data.pkl",'rb')
G,Cls = pickle.load(infile)
infile.close()

n = G.shape[0]
Cl1 = [[i] for i in range(n)]

# nc = 20
# membership = np.array(list(range(nc))*(n//nc+1))[:n]
# np.random.shuffle(membership)

# Cl2 = []
# for i in range(nc):
#     Cl2.append(np.where(membership == i)[0])

h = homophily_effects(G)

betas = [1,2,3]
p = 0.2
r = 10000

data = { "q":[], "beta":[],"clustering":[], "bias":[], "var":[], "var_s":[] }

for beta in betas:
    fY = pom_ugander_yin(G,h,beta)
    TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

    for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=10)(delayed(lambda q : estimate_two_stage(fY,n,p,q,r,beta))(q) for q in np.linspace(p,1,24)):
        data["q"].append(q)
        data["beta"].append(beta)
        data["clustering"].append("everyone")

        mean = np.average(TTE_hat)
        variance = np.average((TTE_hat - mean)**2)
        s_mean = np.average(E_given_U)
        s_variance = np.average((E_given_U - s_mean)**2)

        data["bias"].append(mean - TTE)
        data["var"].append(variance)
        data["var_s"].append(s_variance)

    for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=10)(delayed(lambda q : estimate_two_stage_restricted(fY,n,p,q,r,beta))(q) for q in np.linspace(p,1,24)):
        data["q"].append(q)
        data["beta"].append(beta)

        mean = np.average(TTE_hat)
        variance = np.average((TTE_hat - mean)**2)
        s_mean = np.average(E_given_U)
        s_variance = np.average((E_given_U - s_mean)**2)

        data["bias"].append(mean - TTE)
        data["var"].append(variance)
        data["var_s"].append(s_variance)
        data["clustering"].append("just U")

outfile = open("fig2_data.pkl",'wb')
pickle.dump(data,outfile)
outfile.close()
