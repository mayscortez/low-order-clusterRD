import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle
import time

startTime = time.time()

# parameters
models = ['dyadic','ugander-yin']
betas = [2,2]  # true model degrees, should be same length as models and correspond to its entries
p = 0.1                   # treatment budget
q_values = [0.5, 1]
r = 100                  # number of rct replications
graph_reps = 10          # number of graph replications
n_values = np.linspace(500,5000,10,dtype=int)
Cl = []
estimator_list = ["1-stage","2-stage"]

##############################################

data = {"model":[], "n":[], "q":[], "estimator":[], "bias":[], "var":[]}

def estimate(Cl,n,r,beta,q,model):
    G = er(n,10/n)

    if model == 'dyadic':
        A = G.toarray()
        fY = pom_dyadic(A)
        true_TTE = TTE(fY)
    else:
        h = homophily_effects(G)
        fY = pom_ugander_yin(G,h,beta)
        true_TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

    # 2-stage estimate with true beta
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p,Q,r) 

    tte_hat_2stage = []
    tte_hat_2stage = np.append(tte_hat_2stage, pi_estimate_tte_two_stage(fY(Z),p,Q))

    # 1-stage estimate assuming beta=1
    P = [0,p]
    tte_hat_1stage = []
    tte_hat_1stage = np.append(tte_hat_1stage, one_stage_pi(fY(Z[[0,-1],:,:]), P))

    return (true_TTE, n, tte_hat_1stage, tte_hat_2stage)

# each of these is a dict of dicts of dicts of lists... 
# the outermost dictionary has keys corresponding to the model type
# the value corresponding to each model is itself a dictionary with keys corresponding to estimator ["1-stage","2-stage"]
# the value corresponding to each clustering type is a dictionary with keys corresponding to the n values in n_values
# the value corresponding to each n value is an empty list (to be filled later)
# e.g. Bias_dict[b][e][n] constains a list of the biases of estimator e under a model with true degree b and size n
TTE_hat_dict = {q: {m: {e: {n:[] for n in n_values} for e in estimator_list} for m in models} for q in q_values}
Bias_dict = {q: {m: {e: {n:[] for n in n_values} for e in estimator_list} for m in models} for q in q_values}

for q in q_values:
    print("\nq={}".format(q))
    for idx,model in enumerate(models):
        print("\nmodel={}".format(model))
        beta = betas[idx]
        for g in range(graph_reps):
            if g%5==0:
                print("Graph iteration: {}".format(g))
            for (TTE_true,n,TTE_hat_1stage,TTE_hat_2stage) in Parallel(n_jobs=-2, verbose=5)(delayed(lambda n : estimate([],n,r,beta,q,model))(n) for n in n_values):
                #print("n: {}\nTTE_true: {}\nTTE_hat_1stage: {}\nTTE_hat_2stage:{}".format(n, TTE_true, TTE_hat_1stage,TTE_hat_2stage))
                Bias_dict[q][model]["1-stage"][n].append(TTE_hat_1stage - TTE_true)
                TTE_hat_dict[q][model]["1-stage"][n].append(TTE_hat_1stage)

                Bias_dict[q][model]["2-stage"][n].append(TTE_hat_2stage - TTE_true)
                TTE_hat_dict[q][model]["2-stage"][n].append(TTE_hat_2stage)

# save the data (?)
print("\nSaving the data...")
for q in q_values:
    for model in models:
        for est in ["1-stage","2-stage"]:
            for n in n_values:
                #print(n, est, beta)
                data["n"].append(n)
                data["estimator"].append(est)
                data["model"].append(model)
                data["q"].append(q)

                bias = np.average(Bias_dict[q][model][est][n])
                data["bias"].append(bias)

                var = np.average((TTE_hat_dict[q][model][est][n] - np.average(TTE_hat_dict[q][model][est][n]))**2)
                data["var"].append(var)
                #print("bias: {}\nvar: {}\n".format(bias, var))

#print("data: {}\n".format(data))
file = open("robustness.pkl", "wb")
pickle.dump((data), file)
file.close()

executionTime = (time.time() - startTime)
print('Total runtime in minutes: {}'.format(executionTime/60)) 
        

        