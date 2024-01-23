# This file is to run the experiments where we vary the homophil levels and look at robustness under model misspecfication
from main_vary_phi import main
import numpy as np
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''
theta_prop = 0.5
theta_num = 5
theta = 12 # for deg3, 34; for deg 4, 42
models = [{'type': 'ppom', 'degree':1, 'name':'ppom1', 'params': []},
            {'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []},
            {'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []},
            {'type': 'threshold', 'degree': 2, 'name': 'threshold_prop_' + str(theta_prop).replace(".", ""), 'params': [theta_prop, 'prop']},
            {'type': 'threshold', 'degree': 2, 'name': 'threshold_num_' + str(theta_num), 'params': [theta_num, 'num']},
            {'type': 'saturation', 'degree': 2, 'name': 'saturation_' + str(theta), 'params': [theta]}]
Piis = [0.5, 0.01]
Pijs = [0, 0.01]
probs = [0.06, 0.12, 0.25, 1/3, 2/3, 1]
'''
theta_prop = 0.5
theta_num = 5
theta = 12
models = [{'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []},
            {'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []},
            {'type': 'threshold', 'degree': 2, 'name': 'threshold_prop_' + str(theta_prop).replace(".", ""), 'params': [theta_prop, 'prop']},
            {'type': 'threshold', 'degree': 2, 'name': 'threshold_num_' + str(theta_num), 'params': [theta_num, 'num']},
            {'type': 'saturation', 'degree': 2, 'name': 'saturation_' + str(theta), 'params': [theta]}]
B = 0.06
Piis = [0.5]
probs = [0.06, 0.25, 1]
design = "bernoulli"  # bernoulli   complete
graphNum = 30 
T = 30

for i in range(len(models)):
    print()
    print('==========================================================================')
    print('==========================================================================')
    print('Misspecification, vary Phi experiments for model: {} ({} design)'.format(models[i]["name"], design))
    print('==========================================================================')
    print('==========================================================================')
    for j in range(len(Piis)):
        print('\np_in = {}, p_out = {}\n'.format(Piis[j], np.round((0.5-Piis[j])/49, 3)))
        for p in probs:
            print('p = {}'.format(p))
            print('--------------------------------------------')
            main(models[i], graphNum, T, B, p, Piis[j], design)