import pickle
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(ax,df,color,est):
    ax.plot(df['p'], df['bias'], color=color, label=est)
    ax.fill_between(df['p'], df['bias']-df['sd'], df['bias']+df['sd'], color=color, alpha=0.2)

def draw_plots(data, col_var, row_var, outfile):
    df = pd.DataFrame(data)
    df['sd'] = (df['var'])**(1/2)

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

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    sns.set_theme()

    f,ax = plt.subplots(nrow,ncol, sharex=True, sharey=True)
    plt.setp(ax,xlim=(min(df['p']),max(df['p'])))
    plt.setp(ax,ylim=(-2,1.5))

    ests = df["est"].unique()

    for e,est in enumerate(ests):
        #if est == "hajek": continue

        if ncol == 1:
            plot(ax,df[df["est"] == est],colors[e],est)
        else:    
            for j in range(ncol):
                if nrow == 1:
                    plot(ax[j],df[(df["est"] == est) & (df[col_var] == cols[j])],colors[e],est)
                else:
                    ax[0,j].set_title("{}={}".format(col_var,cols[j]))
                    for i in range(nrow):
                        plot(ax[i,j],df[(df["est"] == est) & (df[row_var] == rows[i]) & (df[col_var] == cols[j])],colors[e],est)
    
    if nrow != 1:
        for i in range(nrow):
            ax[i,0].set_ylabel("{}={}".format(row_var,rows[i]))

    plt.legend()
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

    draw_plots(data, args.col, args.row, args.outfile)
    