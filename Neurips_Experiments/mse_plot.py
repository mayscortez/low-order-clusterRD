import pickle
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# basic mse:            col = nc              row = beta
# compare clustering:   col = clustering      row = nc
# vary p:               col = p               row = nc 
# compare estimators:   col = beta            row = cl

def plot(data, col_var, row_var, outfile):
    '''
    data (dict): experiment data
    col_var (str): 'nc', 'clustering', 'p', or 'beta' or None depending on the experiment
    row_var (str): 'beta', 'nc', or 'cl' or None depending on the experiment
    outfile (str): path to save plot to
    '''
    df = pd.DataFrame(data)

    sns.set_theme()

    # determine rows and columns of subplots
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

    f,ax = plt.subplots(nrow,ncol, sharex=True, sharey=True) #if nrow=1=ncol, just single plot; otherwise, grid of plots
    plt.setp(ax, xlim=(min(df['q']),1))
    plt.setp(ax, ylim=(0,4))
    f.set_figheight(10)
    f.set_figwidth(10)

    if ncol == 1:
        ax.fill_between(df['q'], 0, df['bias']**2, color=colors[0], hatch='++', alpha=0.15, label="Bias$^2$")
        ax.plot(df['q'], df['bias']**2, color=colors[0],alpha=0.5)

        ax.fill_between(df['q'], df['bias']**2, df['bias']**2 + df['var_s'], color=colors[1], alpha=0.15,label="Sampling Variance")
        ax.plot(df['q'], df['bias']**2 + df['var_s'], color=colors[1],alpha=0.5)

        ax.fill_between(df['q'], df['bias']**2 + df['var_s'], df['bias']**2 + df['var'], color=colors[2], hatch='\\\\', alpha=0.15, label="Extrapolation Variance")
        ax.plot(df['q'], df['bias']**2 + df['var'], color='k', linewidth=2, label="MSE")
    else:    
        for j in range(ncol):
            if nrow == 1:
                cell_df = df[(df[col_var] == cols[j])]
                ax[j].set_title("{}={}".format(col_var,cols[j]))
                ax[j].fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], hatch='++', alpha=0.15, label="Bias$^2$")
                ax[j].plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)

                ax[j].fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1], alpha=0.15, label="Sampling Variance")
                ax[j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)

                ax[j].fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], hatch='\\\\', alpha=0.15, label="Extrapolation Variance")
                ax[j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color='k', linewidth=2, label="MSE")
            else:
                ax[0,j].set_title("{}={}".format(col_var,cols[j]))
                for i in range(nrow):
                    cell_df = df[(df[row_var] == rows[i]) & (df[col_var] == cols[j])]
                    ax[i,j].fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], hatch='++', alpha=0.15, label="Bias$^2$")
                    ax[i,j].plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)

                    ax[i,j].fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1], alpha=0.15, label="Sampling Variance")
                    ax[i,j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)

                    ax[i,j].fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], color=colors[2], hatch='\\\\', alpha=0.15, label="Extrapolation Variance")
                    ax[i,j].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color='k', linewidth=2, label="MSE")
    
    if nrow != 1:
        for i in range(nrow):
            ax[i,0].set_ylabel("{}={}".format(row_var,rows[i]))
            #ax[i,0].set_ylabel("{}".format(rows[i]))
    f.supxlabel("effective treatment budget (q)")
    #f.supylabel("clustering")
    #f.suptitle("SBM, Ugander-Yin Potential Outcomes")
    
    ax[0,0].legend(prop={'size':12})
    plt.tight_layout()

    plt.show()
    f.savefig(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',help="path to .pkl file that stores experiment data")
    parser.add_argument('outfile', default="bias_var_plot.png", help="path to save the plot to")
    parser.add_argument('-r','--row',help="basic MSE: row = beta\ncompare clustering: row = nc\nvary p: row = nc")
    parser.add_argument('-c','--col',help="basic MSE: col = nc\ncompare clustering: col = clustering\nvary p: col = p")
    args = parser.parse_args()

    data_file = open(args.infile, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    plot(data, args.col, args.row, args.outfile)
    