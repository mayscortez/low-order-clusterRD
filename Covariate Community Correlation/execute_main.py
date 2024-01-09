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
beta = [1]
B = [0.06]
probs = [[1]] 
design = "bernoulli" # complete   bernoulli
graphNum = 5   
T = 5
for b in range(len(beta)):
    print('=====================================')
    print('Experiments for degree: {}'.format(b+1))
    print('=====================================')
    for p in probs[b]:
        print('B = {}, p = {}'.format(B[b], p))
        print()
        main(beta[b], graphNum, T, B[b], p, design)