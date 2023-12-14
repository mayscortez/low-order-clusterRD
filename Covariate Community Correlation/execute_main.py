# This file is to run the experiments where we vary the correlation between treatment effect type and community type
from main_correlation import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

beta = [1]
B = 0.5
#probs = [1, 5/7, 0.5, 0.3, 0.25, 0.2, 5/35, 0.1]
#probs = [1, 25/30, 25/35, 0.625, 25/45, 0.5]
probs = [1,B]
graphNum = 5   
T = 5
for b in beta:
    for p in probs:
        main(b, graphNum, T, B, p)