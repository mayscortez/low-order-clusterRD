import pickle
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# basic plot:            col = nc, row = beta
# compare clustering:    col = clustering, row = nc

def plot(data, col_var, row_var, include_sample_sd, outfile):
    df = pd.DataFrame(data)
    df['sd'] = (df['var'])**(1/2)
    if include_sample_sd: df['sd_s'] = (df['var_s'])**(1/2)

    colors = ['#0296fb', '#e20287']

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

    f,ax = plt.subplots(nrow,ncol, sharex=True, sharey=True)

    if ncol == 1:
        ax.plot(df['q'], df['bias'], color=colors[1] if include_sample_sd else colors[0])
        ax.fill_between(df['q'], df['bias']-df['sd'], df['bias']+df['sd'], color=colors[0], alpha=0.2)
        ax.fill_between(df['q'], df['bias']-df['sd'], df['bias']+df['sd'], color=colors[0], alpha=0.2)
        if include_sample_sd:
            ax.fill_between(df['q'], df['bias']-df['sd_s'], df['bias']+df['sd_s'], color=colors[1], alpha=0.2) 
    else:    
        for j in range(ncol):
            if nrow == 1:
                cell_df = df[(df[col_var] == cols[j])]
                ax[j].set_title("{}={}".format(col_var,cols[j]))
                ax[j].plot(cell_df['q'], cell_df['bias'], color=colors[1] if include_sample_sd else colors[0])
                ax[j].fill_between(cell_df['q'], cell_df['bias']-cell_df['sd'], cell_df['bias']+cell_df['sd'], color=colors[0], alpha=0.2)
                if include_sample_sd:
                    ax[j].fill_between(cell_df['q'], cell_df['bias']-cell_df['sd_s'], cell_df['bias']+cell_df['sd_s'], color=colors[1], alpha=0.2) 
            else:
                ax[0,j].set_title("{}={}".format(col_var,cols[j]))
                for i in range(nrow):
                    cell_df = df[(df[row_var] == rows[i]) & (df[col_var] == cols[j])]
                    ax[i,j].plot(cell_df['q'], cell_df['bias'], color=colors[1] if include_sample_sd else colors[0])
                    ax[i,j].fill_between(cell_df['q'], cell_df['bias']-cell_df['sd'], cell_df['bias']+cell_df['sd'], color=colors[0], alpha=0.2)
                    if include_sample_sd:
                        ax[i,j].fill_between(cell_df['q'], cell_df['bias']-cell_df['sd_s'], cell_df['bias']+cell_df['sd_s'], color=colors[1], alpha=0.2) 
    
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
    p.add_argument('-s',action='store_true') 
    args = p.parse_args()

    data_file = open(args.infile, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    plot(data, args.col, args.row, args.s, args.outfile)