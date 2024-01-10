# This file is to run the experiments where we vary the treatment probability within clusters and look at robustness under model misspecfication
from main_vary_p import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''
Piis = [0.5, 0.01]
Pijs = [0, 0.01] 
'''
models = [{'type': 'ppom', 'degree':1, 'name':'ppom1', 'params': []},
            {'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []}]
B = 0.06
Piis = [0.5, 0.01]
Pijs = [0, 0.01]  
phis = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
design = "bernoulli"  # bernoulli   complete
graphNum = 30 
T = 30

for i in range(len(models)):
    print()
    print('==========================================================================')
    print('==========================================================================')
    print('Misspecification, vary p experiments for model: ppom{} ({} design)'.format(i+1,design))
    print('==========================================================================')
    print('==========================================================================')
    for j in range(len(Piis)):
        print('\np_in = {}, p_out = {}\n'.format(Piis[j], Pijs[j]))
        for phi in phis:
            print('phi = {}'.format(phi))
            print('--------------------------------------------')
            main(models[i], graphNum, T, B, phi, Piis[j], Pijs[j], design)