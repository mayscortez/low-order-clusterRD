import pickle
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(ax,df,color,est,mse):
    if mse:
        ax.plot(df['p'], df['mse'], color=color, label=est)
    else:
        ax.plot(df['p'], df['bias'], color=color)#, label=est)
        ax.fill_between(df['p'], df['bias']-df['sd'], df['bias']+df['sd'], color=color, alpha=0.2)

def draw_plots(data,outfile,mse):
    df = pd.DataFrame(data)
    df['sd'] = (df['var'])**(1/2)
    df['mse'] = df['mse'] = df['bias']**2 + df['var']

    df = df[(df['treatment']=='cluster') | (df['est'] == 'pi')]
    df = df[(df['est']!='ht')]
    df['est'] = df.apply( lambda row: row['est'] if row['treatment']=='cluster' else 'pi1', axis=1)

    est_names = {
        'pi' : '2-Stage',
        'pi1' : 'PI',
        'dm' : 'DM',
        'dmt' : 'DM(0.75)',
        'hajek' : 'Hajek'
    }

    df['est'] = df.apply( lambda row: est_names[row['est']], axis=1)

    sns.set_theme()

    betas = [1,2]

    colors = ["tab:blue","tab:purple","tab:orange","tab:red","tab:green"]

    sns.set_theme()

    f,ax = plt.subplots(1,len(betas))
    plt.setp(ax,xlim=(min(df['p']),max(df['p'])))
    plt.setp(ax,ylim=(-0.8,0.6))
    plt.setp(ax,xlabel='p')
    plt.setp(ax[0],ylabel='MSE' if mse else 'bias')

    ests = df["est"].unique()

    for e,est in enumerate(ests):
        #if est == "ht": continue
  
        for j,beta in enumerate(betas):
            ax[j].set_title(f"$\\beta={beta}$")
            plot(ax[j],df[(df["est"] == est) & (df["beta"] == beta)],colors[e],est,mse)

    #plt.legend()
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
    