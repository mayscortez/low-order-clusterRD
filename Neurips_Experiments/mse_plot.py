import pickle
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# basic mse:            col = nc              row = beta
# compare clustering:   col = clustering      row = nc
# vary p:               col = p               row = nc 

def plot(data, col_var, row_var, outfile):
    df = pd.DataFrame(data)

    sns.set_theme()

    if row_var != None:
        rows = df[row_var].unique()
        nrow = len(rows)
    else: 
        nrow = 1

    if col_var != None:
        cols = df[col_var].unique()
        ncol = len(cols)
    else:
        ncol = 1

    colors = ["tab:blue","tab:orange","tab:green"]

    f,ax = plt.subplots(nrow,ncol, sharex=True, sharey=True)
    plt.setp(ax, xlim=(min(df['q']),1))
    plt.setp(ax, ylim=(0,1))

    if ncol == 1:
        ax.fill_between(df['q'], 0, df['bias']**2, color=colors[0], alpha=0.15)
        ax.plot(df['q'], df['bias']**2, color=colors[0],alpha=0.5)
        ax.fill_between(df['q'], df['bias']**2, df['bias']**2 + df['var_s'], color=colors[1], alpha=0.15)
        ax.plot(df['q'], df['bias']**2 + df['var_s'], color=colors[1],alpha=0.5)
        ax.fill_between(df['q'], df['bias']**2 + df['var_s'], df['bias']**2 + df['var'], color=colors[2], alpha=0.15)
        ax.plot(df['q'], df['bias']**2 + df['var'], color=colors[2],alpha=0.5)
    else:    
        for j in range(ncol):
            if nrow == 1:
                cell_df = df[(df[col_var] == cols[j])]
                ax[j].set_title("{}={}".format(col_var,cols[j]))
                ax[j].fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], alpha=0.15)
                ax[j].plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
                ax[j].fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1], alpha=0.15)
                ax[j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)
                ax[j].fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], color=colors[2], alpha=0.15)
                ax[j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color=colors[2],alpha=0.5)
            else:
                ax[0,j].set_title("{}={}".format(col_var,cols[j]))
                for i in range(nrow):
                    cell_df = df[(df[row_var] == rows[i]) & (df[col_var] == cols[j])]
                    ax[i,j].fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], alpha=0.15)
                    ax[i,j].plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
                    ax[i,j].fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1], alpha=0.15)
                    ax[i,j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)
                    ax[i,j].fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], color=colors[2], alpha=0.15)
                    ax[i,j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color=colors[2],alpha=0.5)
    
    if nrow != 1:
        for i in range(nrow):
            ax[i,0].set_ylabel("{}={}".format(row_var,rows[i]))

    plt.show()
    f.savefig(outfile)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('infile')
    p.add_argument('outfile', default="bias_var_plot.png")
    p.add_argument('-r','--row')
    p.add_argument('-c','--col')
    args = p.parse_args()

    data_file = open(args.infile, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    plot(data, args.col, args.row, args.outfile)
    