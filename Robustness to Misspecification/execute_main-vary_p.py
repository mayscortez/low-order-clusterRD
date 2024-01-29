# This file is to run the experiments where we vary the treatment probability within clusters and look at robustness under model misspecfication
from main_vary_p import main
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
phis = [0, 0.1, 0.2, 0.3, 0.4, 0.5] 
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
budget = 0.06
Piis = [0.01]
phis = [0,0.5] 
design = "bernoulli"  # bernoulli   complete
graphNum = 25 
T = 5
U = 25

for i in range(len(models)):
    print()
    print('==========================================================================')
    print('==========================================================================')
    print('Misspecification, vary p experiments for model: {} ({} design)'.format(models[i]["name"],design))
    print('==========================================================================')
    print('==========================================================================')
    for j in range(len(Piis)):
        print('\np_in = {}, p_out = {}\n'.format(Piis[j], np.round((0.5-Piis[j])/49, 3)))
        for phi in phis:
            print('phi = {}'.format(phi))
            print('--------------------------------------------')
            main(models[i], graphNum, T, U, budget, phi, Piis[j], design)