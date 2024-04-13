import pickle
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# basic plot:            col = nc, row = beta
# compare clustering:    col = clustering, row = nc

file = open('../Experiments/compare_estimators.pkl', 'rb')
data = pickle.load(file)
file.close()

df = pd.DataFrame(data)

colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

sns.set_theme()

rows = df["beta"].unique()
nrow = len(rows)

cols = df["nc"].unique()
ncol = len(cols)

f,ax = plt.subplots(nrow,ncol, sharex=True, sharey=True)
plt.setp(ax,ylim=(-5,5))

for i in range(nrow):
    ax[i,0].set_ylabel("beta={}".format(rows[i]))
    for j in range(ncol):
        cell_df = df[(df["beta"] == rows[i]) & (df["nc"] == cols[j])]

        for k,est in enumerate(cell_df["est"].unique()):
            line_df = cell_df[(cell_df["est"] == est)]
            ax[i,j].plot(line_df['q'], line_df['bias'], color=colors[k])
            ax[i,j].fill_between(line_df['q'], line_df['bias']-line_df['sd'], line_df['bias']+line_df['sd'], color=colors[k], alpha=0.2)

for j in range(ncol):
    ax[0,j].set_title("nc={}".format(cols[j]))

plt.show()
f.savefig("compare_estimators.png")