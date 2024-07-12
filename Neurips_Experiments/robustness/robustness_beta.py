import sys
sys.path.insert(0, "../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle
import time

startTime = time.time()

# parameters
betas = [2,3,4]             # model degrees
p = 0.1                   # treatment budget
q_values = [0.1, 0.5, 1]
r = 100                  # number of rct replications
graph_reps = 10          # number of graph replications
n_values = np.linspace(500,5000,10,dtype=int)
Cl = []
estimator_list = ["1-stage","2-stage"]

##############################################

data = {"beta":[], "n":[], "q":[], "estimator":[], "bias":[], "var":[]}#, "var_s":[]}

def estimate(Cl,n,r,beta,q):
    G = er(n,10/n)
    h = homophily_effects(G)
    fY = pom_ugander_yin(G,h,beta)
    true_TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

    # 2-stage estimate with true beta
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p,Q,r) 

    tte_hat_2stage = []
    tte_hat_2stage = np.append(tte_hat_2stage, pi_estimate_tte_two_stage(fY(Z),p,Q))
    #e_tte_hat_given_u_2stage = []
    #e_tte_hat_given_u_2stage = np.append(e_tte_hat_given_u_2stage, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    # 1-stage estimate assuming beta=1
    P = [0,p]
    tte_hat_1stage = []
    tte_hat_1stage = np.append(tte_hat_1stage, one_stage_pi(fY(Z[[0,-1],:,:]), P))
    #e_tte_hat_given_u_1stage = []
    #e_tte_hat_given_u_1stage = np.append(e_tte_hat_given_u_1stage, )

    return (true_TTE, n, tte_hat_1stage, tte_hat_2stage)

# each of these is a dict of dicts of dicts of lists... 
# the outermost dictionary has keys corresponding to the model degrees (betas)
# the value corresponding to each beta is itself a dictionary with keys corresponding to estimator ["1-stage","2-stage"]
# the value corresponding to each clustering type is a dictionary with keys corresponding to the n values in n_values
# the value corresponding to each n value is an empty list (to be filled later)
# e.g. Bias_dict[b][e][n] constains a list of the biases of estimator e under a model with true degree b and size n
TTE_hat_dict = {q: {b: {e: {n:[] for n in n_values} for e in estimator_list} for b in betas} for q in q_values}
Bias_dict = {q: {b: {e: {n:[] for n in n_values} for e in estimator_list} for b in betas} for q in q_values}
#E_given_U_dict = {b: {e: {n:[] for n in n_values} for e in estimator_list} for b in betas}

for q in q_values:
    print("\nq={}".format(q))
    for beta in betas:
        print("\nbeta={}".format(beta))
        for g in range(graph_reps):
            if g%5==0:
                print("Graph iteration: {}".format(g))
            for (TTE_true,n,TTE_hat_1stage,TTE_hat_2stage) in Parallel(n_jobs=-2, verbose=5)(delayed(lambda n : estimate([],n,r,beta,q))(n) for n in n_values):
                #print("n: {}\nTTE_true: {}\nTTE_hat_1stage: {}\nTTE_hat_2stage:{}".format(n, TTE_true, TTE_hat_1stage,TTE_hat_2stage))
                Bias_dict[q][beta]["1-stage"][n].append(TTE_hat_1stage - TTE_true)
                TTE_hat_dict[q][beta]["1-stage"][n].append(TTE_hat_1stage)
                #E_given_U_dict[beta]["1-stage"][n].append(E_given_U_1stage)

                Bias_dict[q][beta]["2-stage"][n].append(TTE_hat_2stage - TTE_true)
                TTE_hat_dict[q][beta]["2-stage"][n].append(TTE_hat_2stage)
                #E_given_U_dict[beta]["2-stage"][n].append(E_given_U_2stage)

# save the data (?)
print("\nSaving the data...")
for q in q_values:
    for beta in betas:
        for est in ["1-stage","2-stage"]:
            for n in n_values:
                #print(n, est, beta)
                data["n"].append(n)
                data["estimator"].append(est)
                data["beta"].append(beta)
                data["q"].append(q)

                bias = np.average(Bias_dict[q][beta][est][n])
                data["bias"].append(bias)

                var = np.average((TTE_hat_dict[q][beta][est][n] - np.average(TTE_hat_dict[q][beta][est][n]))**2)
                data["var"].append(var)
                #print("bias: {}\nvar: {}\n".format(bias, var))
                #data["var_s"].append(np.average((E_given_U_dict[beta][cl][q] - np.average(E_given_U_dict[beta][cl][q]))**2))

#print("data: {}\n".format(data))
file = open("sbm_robustness.pkl", "wb")
pickle.dump((data), file)
file.close()

executionTime = (time.time() - startTime)
print('Total runtime in minutes: {}'.format(executionTime/60)) 
        

        