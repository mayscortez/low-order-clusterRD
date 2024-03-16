import pickle
import re
import numpy as np

G_dict = {} # dictionary mapping each vertex to list of neighbors

users = set()

for line in open("edges.txt", "r"):
    m = re.match("^([0-9]*) ([0-9]*)$",line)
    u,v = m.group(1,2)
    u,v = int(u),int(v)

    users.add(u)
    users.add(v)

    if v not in G_dict:
        G_dict[v] = [u]      # we want to keep track of in-neighbors, so this is "backwards"
    else:
        G_dict[v].append(u)

n = len(users)

for i in range(n):
    if i not in G_dict:
        G_dict[i] = [i]
    else:
        G_dict[i].append(i)
        G_dict[i] = list(set(G_dict[i])) # remove duplicates

G = []
for i in range(n):
    G.append(sorted(G_dict[i]))

membership = np.empty(n)

for line in open("communities.txt", "r"):
    m = re.match("^([0-9]*) ([0-9]*)$",line)
    v,c = m.group(1,2)
    v,c = int(v),int(c)

    membership[v] = c

Cl = []
for i in range(int(np.max(membership))+1):
    Cl.append(list(np.where(membership == i)[0]))

file = open("graph_data.pkl", "wb")
pickle.dump((G,Cl,membership), file)
file.close()
