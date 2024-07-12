import matplotlib.pyplot as plt
import numpy as np

def plot_robustness(df, col_var, row_var, model, x_var, plot_type="bias", save_path=""):  
    '''
    Plot the results from robustness_beta.py as a grid of plots

    col_var (str) : variable from df you want the columns of the plot to correspond to
    row_var (str) : variable from df you want the rows of the plot to correspond to
    model (str) : name of potential outcomes model (e.g. Ugander-Yin)
    x_var (str) : variable from df you want on the x-axis for each subplot
    plot_type (str) : either "bias" or "mse" 
    save_path (str) : where you want to save the resulting image
    '''  
    cols = df[col_var].unique()
    rows = df[row_var].unique()
    ncol = len(cols)
    nrow = len(rows)

    f,ax = plt.subplots(nrow,ncol, sharex=True, sharey=True)
    plt.setp(ax, xlim=(min(df['n']),max(df['n'])))
    plt.setp(ax, ylim=(-10,10))
    f.suptitle(model)
    f.set_figheight(10)
    f.set_figwidth(10)
    f.supxlabel(x_var)

    for j in range(ncol):
        ax[0,j].set_title("{}={}".format(col_var,cols[j]))
        for i in range(nrow):
            ax[i,0].set_ylabel("{}={}".format(row_var,rows[i]))
            for x in df["estimator"].unique():
                sub_df = df[(df[row_var] == rows[i]) & (df[col_var] == cols[j]) & (df["estimator"] == x)]
                if plot_type == "mse":
                    ax[i,j].plot(sub_df[x_var], sub_df['bias']**2 + sub_df['var'], label=x)
                else:
                    ax[i,j].plot(sub_df[x_var], sub_df['bias'], label=x)
                    sd = np.sqrt(sub_df['var'])
                    ax[i,j].fill_between(sub_df[x_var],  sub_df['bias']-sd, sub_df['bias']+sd, alpha=0.15)
    ax[0,0].legend()
    plt.show()
    f.savefig(save_path)