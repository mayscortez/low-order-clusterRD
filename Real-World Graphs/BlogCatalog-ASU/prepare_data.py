import pickle
import re

G_dict = {} # dictionary mapping each vertex to list of neighbors

for line in open("edges.csv", "r"):
    m = re.match("^([0-9]*),([0-9]*)$",line)
    u,v = m.group(1,2)
    u,v = int(u)-1,int(v)-1

    if u not in G_dict:
        G_dict[u] = [v]
    else:
        G_dict[u].append(v)

    if v not in G_dict:
        G_dict[v] = [u]
    else:
        G_dict[v].append(v)

n = len(G_dict.keys())
for i in range(n):
    G_dict[i].append(i)

G = []
for i in range(n):
    G.append(sorted(G_dict[i]))

features = { i:[] for i in range(n)} # dictionary mapping each vertex to list of communities

for line in open("communities.csv", "r"):
    m = re.match("^([0-9]*),([0-9]*)$",line)
    v,c = m.group(1,2)
    v,c = int(v)-1,int(c)-1

    features[v].append(c)

file = open("graph_data.pkl", "wb")
pickle.dump((G,features), file)
file.close()
