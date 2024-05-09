import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot(ax,df,mse,**kwargs):
    if mse:
        ax.plot(df['p'], df['mse'],**kwargs)
    else:
        ax.plot(df['p'], df['bias'],**kwargs)

        kwargs["label"]=None
        if "marker" in kwargs: kwargs.pop("marker")
        if "markersize" in kwargs: kwargs.pop("markersize")
        if "linestyle" in kwargs: kwargs.pop("linestyle")
        ax.fill_between(df['p'], df['bias']-df['sd'], df['bias']+df['sd'],alpha=0.2, **kwargs)

def draw_plots(data,outfile,mse):
    df = pd.DataFrame(data)
    df['sd'] = (df['var'])**(1/2)
    df['mse'] = df['mse'] = df['bias']**2 + df['var']

    df = df[(df['treatment']=='cluster') | (df['est'] == 'pi')]
    df = df[(df['est']!='ht')]
    df['est'] = df.apply( lambda row: row['est'] if row['treatment']=='cluster' else 'pi1', axis=1)

    colors = ["tab:blue","tab:purple","tab:orange","tab:red","tab:green"]

    est_kws = {
        'pi' : {
            'label': '2-Stage',
            'color': colors[0],
            'marker': 'o',
            'markersize': 4
        },
        'pi1' : {
            'label': 'PI',
            'color': colors[1],
            'linestyle': '--'
        },
        'dm' : {
            'label': 'DM',
            'color': colors[2],
            'marker': 'x',
            'markersize': 6
        },
        'dmt' : {
            'label': 'DM(0.75)',
            'color': colors[3],
            'linestyle': '-.'
        },
        'hajek' : {
            'label': 'Hájek',
            'color': colors[4],
            'linestyle': 'dotted'
        }
    }

    #marks = ['D','*','s','x','o']

    betas = [1,2]

    f,ax = plt.subplots(1,len(betas))

    plt.setp(ax,xlim=(min(df['p']),max(df['p'])))

    # Amazon
    # if mse:
    #     plt.setp(ax,ylim=(0,0.5))
    # else:
    #     plt.setp(ax,ylim=(-0.8,0.6))

    # Email
    # if mse:
    #     plt.setp(ax,ylim=(0,12))
    # else:
    #     plt.setp(ax,ylim=(-5,3))

    # BlogCatalog
    if mse:
        plt.setp(ax,ylim=(0,9))
    else:
        plt.setp(ax,ylim=(-4,2))
    
    
    for a in ax:
        a.set_xlabel('p',fontsize=14)
        a.set_ylabel('MSE' if mse else 'Bias', fontsize=14)

    ests = df["est"].unique()

    for est in ests:
        if est == 'hajek': continue
        for j,beta in enumerate(betas):
            ax[j].set_title(f"$\\beta={beta}$", fontsize=16)
            plot(ax[j],df[(df["est"] == est) & (df["beta"] == beta)],mse,**est_kws[est])

    ax[0].legend(ncol=2,prop={'size': 12})
    f.subplots_adjust(bottom=0.25)
    plt.show()
    f.savefig(outfile)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('infile')
    p.add_argument('outfile', default="compare_estimators_plot.png")
    p.add_argument('-m','--mse',action='store_true')
    args = p.parse_args()

    data_file = open(args.infile, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    draw_plots(data,args.outfile,args.mse)
    