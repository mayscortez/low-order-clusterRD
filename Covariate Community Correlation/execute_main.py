# This file is to run the experiments where we vary the correlation between treatment effect type and community type
from main_correlation import main
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
beta = [1,2]
B = [0.02, 0.02] 
probs = [[0.02, 0.1, 0.2, 0.3, 1], [0.02, 0.1, 0.2, 0.3, 1]] # K = 1, 5, 10, 15, 50
designs = ["bernoulli"] # complete   bernoulli
graphNum = 30  
T = 30
for d in designs:
    for b in range(len(beta)):
        print('======================================================================================')
        print('Homophily experiments for degree: {} ({} design)'.format(b+1, d))
        print('======================================================================================')
        for p in probs[b]:
            print('B = {}, p = {}'.format(B[b], p))
            print()
            main(beta[b], graphNum, T, B[b], p, d)