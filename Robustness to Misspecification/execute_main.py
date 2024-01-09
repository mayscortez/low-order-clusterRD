# This file is to run the experiments where we vary the correlation between treatment effect type and community type
from main_misspecification import main
import numpy as np
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''
beta = [1,2]
B = [0.06, 0.5]
probs = [[0.06, 0.25, 1/3, 2/3, 1],     # K in [50, 12, 9, 6, 3]
         [0.5, 0.625, 25/33, 25/29, 1]]#, # K in [50, 40, 33, 29, 25]
'''
models = [{'type': 'ppom', 'degree':1, 'name':'ppom1', 'params': []},
            {'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []}]
B = 0.06
Piis = [0.5, 0.01]
Pijs = [0, 0.01]  
graphNum = 30 
T = 30

for i in range(len(models)):
    print()
    print('==========================================================================')
    print('==========================================================================')
    print('Experiments for model: ppom{}'.format(i+1))
    print('==========================================================================')
    print('==========================================================================')
    for j in range(len(Piis)):
        print('p_in = {}, p_out = {}'.format(Piis[j], Pijs[j]))
        print()
        main(models[i], graphNum, T, B, Piis[j], Pijs[j])