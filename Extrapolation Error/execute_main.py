# This file is to run the experiments where we vary the correlation between treatment effect type and community type
from main_extrapolation import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''
beta = [1,2,3]
B = 0.06
phis = [0, 0.25, 0.5]
'''
beta = [1,2,3]
B = 0.06 
phis = [0, 0.25, 0.5]
design = "bernoulli" # options: "complete" or "bernoulli"
p_in = 0.5
graphNum = 30  
T = 30

for b in range(len(beta)):
    print('============================================================================================')
    print('Extrapolation experiments for degree: {} ({} design, p_in = {}, B = {})'.format(b+1, design, p_in, B))
    print('============================================================================================')
    for phi in phis:
        print('phi = {}'.format(phi))
        print()
        main(beta[b], graphNum, T, B, phi, p_in, design)