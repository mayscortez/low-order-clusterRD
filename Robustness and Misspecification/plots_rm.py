# plots
#from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
#pd.options.mode.chained_assignment = None  # default='warn'

def main(model, B=0.06, p_in=0.5, p_out=0, phi=0, p=1, type='MSE'): 
    model_name = model['name']
    degree = model['degree']

    n = 1000
    nc = 50
    Bstr = str(np.round(B,3)).replace('.','')
    
    experiment = 'misspecified'

    fixed = 'n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_B' + Bstr # naming convention
    
    x_label = [fixed + '_' + experiment,
               fixed + '_' + experiment]
    x_var = ['Phi', 'p']
    x_plot = ['$\phi$', '$p$']
    title = ['$\\beta={}, SBM({},{},{},{}), B={}, p={}$'.format(degree, n, nc, np.round(p_in,3), np.round(p_out,3), B, np.round(p,3)),
             '$\\beta={}, SBM({},{},{},{}), B={}, \phi={}$'.format(degree, n, nc, np.round(p_in,3), np.round(p_out,3), B, np.round(phi,3))]
    var_to_fix  = ['p', 'Phi']
    fixed_var_values = [p, phi]
    for ind in range(len(x_var)):
        plot(x_var[ind],x_label[ind],model_name,x_plot[ind],title[ind],var_to_fix[ind],fixed_var_values[ind], type)

def plot(x_var, x_label, model, x_plot, title, fixed_param_name, fixed_param_value, type='MSE'):
    load_path = 'output/' + model + '/'
    save_path = 'plots/' + model + '/'
    
    # Possible estimators: 
        # names_ClRD = ['HT', 'PI-$n(p;1)$', 'PI-$\mathcal{U}(p;1)$', 'LS-PropC(1)', 'LS-NumC(1)', 'PI-$n(p;2)$', 'PI-$\mathcal{U}(p;2)$', 'LS-PropC(2)', 'LS-NumC(2)','PI-$n(p;3)$', 'PI-$\mathcal{U}(p;3)$', 'LS-PropC(3)', 'LS-NumC(3)']
        # names_BRD = ['PI-$n(B;1)$', 'LS-PropB(1)', 'LS-NumB(1)', 'PI-$n(B;2)$', 'LS-PropB(2)', 'LS-NumB(2)', 'PI-$n(B;3)$', 'LS-PropB(3)', 'LS-NumB(3)','DM', 'DM($0.75$)']
    estimators = ['PI-$\mathcal{U}(p;1)$','PI-$\mathcal{U}(p;2)$','PI-$\mathcal{U}(p;3)$', 'PI-$n(B;1)$'] 

    experiment = x_label
    print('\n'+experiment+'_'+'vary-'+x_var)

    # Create and save plots
    #test_str = load_path+model+'_'+experiment+'-full-data' + '.csv'
    #print(test_str)
    
    df = pd.read_csv(load_path+model+'_'+experiment+'-full-data' + '.csv')
    newData = df.loc[(df['Estimator'].isin(estimators)) & (df[fixed_param_name] == fixed_param_value)]

    plt.rc('text', usetex=True)
    
    if (type == 'MSE' or type == 'both'):
        # MSE plots
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_var, y='Rel_bias_sq', hue='Estimator', style='Estimator', data=newData, legend='brief', markers=True)

        #ax.set_xlim(0,0.001)
        ax.set_xlabel(x_plot, fontsize = 18)
        if fixed_param_name == 'Phi':
            ax.set_ylim(-10,20)
        ax.set_ylabel("MSE", fontsize = 18)
        ax.set_title(title, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
        plt.grid()
        plt.tight_layout()

        fix_str = str(np.round(fixed_param_value,3)).replace('.','')
        save_str = save_path + model + experiment + '_vary-' + x_var +'_' + fixed_param_name + fix_str + '_MSE.png'
        plt.savefig(save_str)
        print("saved: " + save_str)
        plt.close()

    if (type == 'Bias' or type == 'both'):
        # Plot with all the estimators
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', errorbar='sd', data=newData, legend='brief', markers=True)

        #ax.set_xlim(0,0.001)
        ax.set_xlabel(x_plot, fontsize = 18)
        if fixed_param_name == 'Phi':
            ax.set_ylim(-10,20)
        ax.set_ylabel("Relative Bias", fontsize = 18)
        ax.set_title(title, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
        plt.grid()
        plt.tight_layout()

        fix_str = str(np.round(fixed_param_value,3)).replace('.','')
        save_str = save_path + model + experiment + '_vary-' + x_var +'_' + fixed_param_name + fix_str + '.png'
        plt.savefig(save_str)
        print("saved: " + save_str)
        plt.close()
    

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    models = [{'type': 'ppom', 'degree':1, 'name':'ppom1', 'params': []},
            {'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []}]
    B = 0.06
    Piis = [0.5, 0.01]
    Pijs = [0, 0.01]
    phi = 0
    p = 1  

for i in range(len(models)):
    print('Plotting for true model: {}'.format(models[i]['name']))
    for j in range(len(Piis)):
        main(models[i], B, Piis[j], Pijs[j], phi, p, 'MSE')
    print() 
