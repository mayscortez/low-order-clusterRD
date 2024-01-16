This README file explains the contents of the folders `Covariate Community Correlation`, `Robustness to Misspecification`, `Increasing Edges`, including how to run experiments and plot results.

For all experiments, the networks are stochastic block models of size $n=1000$, there are $n_c = 50$ communities/blocks of size $20$, and the expected degree is $10$. The parameter $p_{\text{in}}$ tunes the connectivity of the graph and represents the edge probability between units of the same cluster. Given this parameter, the main experiment files automatically compute the edge probability between units of different clusters, $p_{\text{out}}$ such that the expected degree is 10. When $p_{\text{in}}=0.5$, $p_{\text{out}}=0$ and the graph is made of disjoint clusters. When $p_{\text{in}}=0.01$, $p_{\text{out}}=0.01$ and the graph is really Erd\"os-R\'enyi. To change these network details, the main experiment files would need to be modified accordingly.

## Covariate Community Correlation
The folder `Covariate Community Correlation` has the files for the experiment varying homophily levels, while keeping other parameters (such as graph connectivity and treatment probabilities) fixed.
- `main_correlation.py`: main experiment file
- `execute_main.py`: runs the main experiment
- `experiment_functions.py`: helper functions for the experiment
- `plots.py`: script that plots the output from the experiment
- `output`: folder containing the results from the experiments (there is a seperate folder for each model degee)
    - the subfolders `deg1`, `deg2`, `deg3` correspond to different models (with  $\beta=1$, $\beta=2$, and $\beta=3$, respectively.)
    - the subsubfolders `bernoulli` and `complete` denote the design for choosing clusters in the first stage of the two-stage cluster staggered rollout.
- `plots`: folder containing the resulting plots from `plots.py` (there is a seperate folder for each model degee)
    - folder structure is the same as output

### Running Covariate Community Correlation Experiments
The parameter $\phi$ tunes the balance of clusters with respect to covariates. When $\phi=0$, there is no balance, aka covariate type is perfectly correlated with cluster, and therefore high (perfect) homophily. When $\phi=0.5$, covariates are balanced across clusters, meaning there is low to no homophily. In these experiments, we vary $\phi$ between these two extremes. 

To run an experiment and save the output, visit the file `execute_main.py`. The file is capable of running several experiments back to back, saving the corresponding output in the corresponding folders. The 6 parameters determining the details of the experiment are listed and explained below:
- `beta`: list of the model degrees
- `B`: list of treatment budgets
- `probs`: list of lists of treatment probabilities
- `designs`: list of strings
- `p_in`: float - the edge probability between units of the same cluster
- `graphNum`: int - the number of graphs to average over in the experiment
- `T`: int - the number of treatment samples to average over per graph

Note that there should be the same number of elements in `beta`, `B`, and `probs`. When running multiple experiments back to back, the $i$th element of `beta` corresponds to the `i`th element of `B` and `probs`.

**Naming Convention**

There are two outputs per experiment: a text file that records how long the experiment took and a csv file recording the results from the experiment. Each file is prefixed with `correlation_n1000_nc50_` denoting that this is the correlation experiment on a graph with $n=1000$ nodes and $n_c=50$ clusters. The next part of the string conatins information about $p_\text{in}, p_\text{out}, p, B, \beta$ and the design for choosing clusters. Lastly, the csv files end with the suffix `-full.csv` while the text files simply end with `.txt`. For example, `correlation_n1000_nc50_in05_out00_p1_B006_deg1_bernoulli` means $p_\text{in}=0.5, p_\text{out}=0.0, p=1, B=0.06, \beta=1$. Note that decimals are removed from the probabilities, so $0.06$ becomes `006` and $0.6$ becomes `06`.

**Example: Run a Single Experiment**

Suppose you want to run a *single* experiment where the model degree is $\beta=2$, treatment budget is $B=0.06$, treatment probability within chosen clusters is $p=1$, clusters are chosen according to Bernoulli design, we average over 30 SBMs with $p_\text{in}=0.5, p_\text{out}=0$ (disjoint clusters) and for each graph, average over 30 samples. Then the corresponding parameters to set in the `execute_main.py` file are 
- `beta = [2]`
- `B = [0.06]`
- `probs = [[1]]`
- `designs = ["bernoulli"]`
- `p_in = 0.5`
- `graphNum = 30`
- `T = 30`

**Example: Run Multiple Experiments**

Suppose you choose the parameters for `execute_main.py`:
- `beta = [1,2]`
- `B = [0.06, 0.5]`
- `probs = [[0.06, 0.25, 1/3, 2/3, 1], [0.5, 0.625, 25/33, 25/29, 1]]`
- `designs = ["bernoulli"]`
- `p_in = 0.01`
- `graphNum = 30`
- `T = 30`

This runs 5 *seperate* experiments for a degree $\beta=1$ model and 5 *seperate* experiments for a degree $\beta=2$ model. Each averages over 30 SBMs, where $p_\text{in}=0.01=p_\text{out}$ (aka Erdos-Renyi graphs), and 30 treatment samples per graph. The degree $\beta=1$ experiments all have a treatment budget of $B=0.06$, but each corresponds to a different treatment probability $p \in \{0.06, 0.25, 1/3, 2/3, 1\}$. Note this corresponds to choosing $K=50, 12, 9, 6, 3$ clusters out of $50$, respectively. The degree $\beta=2$ experiments all have a treatment budget of $B=0.5$, but each corresponds to a different treatment probability $p \in \{0.5, 0.625, 25/33, 25/29, 1\}$. Note this corresponds to choosing $K=50, 40, 33, 29, 25$ clusters out of $50$, respectively. 

### Plotting Results
The script `plots.py` plots the output from the experiment in `main_correlation.py`
- To run `plots.py`, scroll down to the bottom of the file. 
- The parameters to change, depending on what you'd like to plot, are `beta`, `B`, `probs`, `p_in`, `design`, and `estimators`
- For example, to plot results for the Polynomial Interpolation (PI) estimators under cluster staggered rollout and Bernoulli staggered when the underlying model is a degree 2 polynomial ($\beta=2$), the treatment budget is $B=0.06$, the treatment probability within chosen clusters is $p=1$, cluster selection is Bernoulli design, and the graph is made up of disjoint clusters, the parameters would be:
    - `beta = [2]`
    - `B = [0.06]`
    - `probs = [[1]]`
    - `p_in = 0.5`
    - `design = "bernoulli"` 
    - `estimators = ['PI-$n$($p$)', 'PI-$\mathcal{U}$($p$)', 'PI-$n$($B$)']`
- Suppose you want to plot results for the same estimators, under the same graph model and design for choosing clusters, but you want to generate multiple plots corresponding to different treatment probabilties $p$ and for multiple model degrees $\beta$, i.e. plot the results from multiple seperate experiments. You can modify the parameters as follows:
    - `beta = [1, 2]`
    - `B = [0.06, 0.06]`
    - `probs = [[0.06, 0.25, 1/3, 2/3, 1], [0.06, 0.25, 1/3, 2/3, 1]]`
    - `p_in = 0.5`
    - `design = "bernoulli"` 
    - `estimators = ['PI-$n$($p$)', 'PI-$\mathcal{U}$($p$)', 'PI-$n$($B$)']`

Estimator choices are as follows:
- `'PI-$n$($p$)'` : polynomial interpolation estimator under cluster staggered rollout design that uses all $n$ observations
- `'PI-$\mathcal{U}$($p$)'` : polynomial interpolation estimator under cluster staggered rollout design that only usees observations from chosen clusters
- `'HT'` : Horvitz-Thompson estimator, cluster design
- `'DM-C'` : simple Difference in Means, cluster design
- `'DM-C($0.75$)'` : thresholded Difference in Means, cluster design
- `'PI-$n$($B$)'` : polynomial interpolation estimator under simple Bernoulli staggered rollout design
- `'LS-Prop'` : least squares estimator regressing over proportion of neighbors treated, Bernoulli design
- `'LS-Num'` : least squares estimator regressing over number of neighbors treated, Bernoulli design
- `'DM'` : simple Difference in Means, Bernoulli design
- `'DM($0.75$)'` : thresholded Difference in Means, Bernoulli design

NOTE: When choosing estimators to plot, make sure they remain in the same relative order as the list above to ensure consistency of colors in the plots. 

## Robustness to Misspecification
This folder has the files for the experiments comparing the performance of different estimators and designs under model misspecification. There are **two** different experiments, one where we vary the balance (homophily) level $\phi$ and one where we vary the treatment probability within chosen clusters $p$.
- `main_vary_p.py`: main experiment where we vary $p$
- `execute_main-vary_p.py`: runs the main experiment varying $p$ (this is the file to modify to run your own experiment)
- `main_vary_phi.py`: main experiment where we vary $\phi$
- `execute_main-vary_phi.py`: runs the main experiment varying $\phi$ (this is the file to modify to run your own experiment)
- `experiment_functions.py`: helper functions for the experiment
- `plots_rm_vary_p.py`: script that plots the output from the experiment where we vary $p$
- `plots_rm_vary_phi.py`: script that plots the output from the experiment where we vary $\phi$
- `output` is a folder containing the results from the experiments
    - the subfolders `vary_p` and `vary_phi` contain the results from the respective experiments
    - the subsubfolders `ppom1`, `ppom2`, and `ppom3` correspond to results when the *true* model is polynomial with degree $\beta=1$, $\beta=2$, and $\beta=3$, respectively. If we choose to try other models, such as threshold models, we should create a new folder for those results.
    - the subsubsubfolders `bernoulli` and `complete` denote the design for choosing clusters in the first stage of the two-stage cluster staggered rollout.
- `plots` is a folder containing the resulting plots from `plots.py`
    - folder structure is the same as `output`


### Running Robustness to Misspecification Experiments
In each experiment, a *true* model is fixed. As we vary *either* the treatment probability for chosen clusteres $p$ or the covariate balance/homophily level $\phi$, we compute estimates for the TTE with various estimators, most assuming a misspecified model. The idea is to compare the performance of different estimators under misspecified models to better understand when some perform better than others.

## Varying treatment probability $p$
In this experiment, we fix the true model and vary the treatment probability within chosen clusters $p$ from $p=B$ to $p=1$ and compute the relative bias, absolute bias, and MSE of different estimators assuming misspecified models. 

To run an experiment and save the output, visit the file `execute_main-vary_p.py`. The file is capable of running several experiments back to back, saving the corresponding output in the corresponding folders. The 7 parameters determining the details of the experiment are listed and explained below:
- `models`: list of the potential outcomes models you want to run the experiments for
    - currently, only polynomial potential outcomes models of degrees $\beta=1,2,3,4$ are implemented
    - a model is a dictionary with 4 keys: 
        - `'type'`: e.g. `'ppom'` for polynomial potential outcomes model or `'threshold'` for a threshold model (not yet implemented)
        - `'degree'`: model degree, i.e. $\beta$
        - `'name'`: the name for the model e.g. `'ppom2'` for polynomial potential outcomes model of degree $\beta=2$ 
        - `'params'`: additional parameters specific to the model -- for ppom models this is just an empty list
- `B`: treatment budget
- `Piis`: list of edge probabilities between units of the same cluster
- `phis` : list of values of $\phi$ you want to run the experiment for
- `design`: string denoting which design to use to choose clusters
- `graphNum`: int - the number of graphs to average over in the experiment
- `T`: int - the number of treatment samples to average over per graph

**Example**

Suppose you want to run these experiments where the true models are polynomials with degrees $\beta=3,4$ on a graph with disjoint clusters ($\p_\text{in}=0.5$), and for two cluster balance levels $\phi=0$ (perfect homophily) and $\phi=0.5$ (no homophily) with Bernoulli choice of clusters with a treatment budget of $B=0.06$. The corresponding parameters would be:
- `models = [{'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []}, {'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []}]`
- `B = 0.06`
- `Piis = [0.5]`
- `phis = [0, 0.5]`
- `design = "bernoulli"`
- `graphNum = 30` 
- `T = 30`

## Varying balance/homophily $\phi$
In this experiment, we fix the true model and vary the covariate balance/homphily level within chosen clusters $\phi$ from $\phi=0$ to $\phi=0.5$ and compute the relative bias, absolute bias, and MSE of different estimators assuming misspecified models.

To run an experiment and save the output, visit the file `execute_main-vary_phi.py`. Instructions are very similar to the ones for `execute_main-vary_phi.py` but instead of the parameter `phis` we use the parameter `probs`, a list of treatment probabilites you want to run the experiment for.

### Plotting Results
`plots_rm_vary_p.py` is a script that plots the output from the experiment `main_vary_p`. 
- To run `plots_rm_vary_p.py`, scroll down to the bottom of the file for the experiments you want to plot. 
- The parameters to change, depending on what you'd like to plot, are `models`, `B`, `Piis`, `Pijs`, `phis`, `cluster_selection`, `type`, and `estimators`
- For example, to plot the MSE for the Polynomial Interpolation estimators under cluster staggered rollout and Bernoulli staggered rollout assuming a linear ($\beta=1$) model when the true underlying model is degree 4 polynomial ($\beta=4$), the treatment budget is $B=0.06$, cluster selection is Bernoulli design, clusters are balanced across covariates ($\phi=0.5$), and the graph is made up of disjoint clusters, the parameters would be:
    - `models = [{'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []}]`
    - `B = 0.06`
    - `Piis = [0.5]`
    - `Pijs = [0]`
    - `phis = [0.5]`
    - `cluster_selection = "bernoulli"`
    - `type = "MSE"`
    - `estimators = ['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$']`
- To plot for several different graph structures and balance/homophily levels, note that you can add these all to the lists `Piis`, `Pijs`, `phis`. The code is already set up to iterate over the values in there.

`plots_rm_vary_phi.py` is a script that plots the output from the experiment `main_vary_phi`. 
- The structure is the same as above, except instead of the variable `phis`, you have the variable `probs` corresponding to the treatment probabilities (within clusters) you want to plot for.
- You can plot MSE as a function of $\phi$, Relative Bias as a function of $\phi$, or both. 

**Estimator choices**
- `'PI-$n(p;1)$'` : polynomial interpolation estimator under cluster staggered rollout design that uses all $n$ observations, assuming a degree $\beta=1$ model
- `'PI-$\mathcal{U}(p;1)$'` : polynomial interpolation estimator under cluster staggered rollout design that only usees observations from chosen clusters, assuming a degree $\beta=1$ model
- `'PI-$n(p;2)$'` : polynomial interpolation estimator under cluster staggered rollout design that uses all $n$ observations, assuming a degree $\beta=2$ model
- `'PI-$\mathcal{U}(p;2)$'` : polynomial interpolation estimator under cluster staggered rollout design that only usees observations from chosen clusters, assuming a degree $\beta=2$ model
- `'PI-$n(p;3)$'` : polynomial interpolation estimator under cluster staggered rollout design that uses all $n$ observations, assuming a degree $\beta=3$ model
- `'PI-$\mathcal{U}(p;3)$'` : polynomial interpolation estimator under cluster staggered rollout design that only usees observations from chosen clusters, assuming a degree $\beta=3$ model
- `'HT'` : Horvitz-Thompson estimator, cluster design
- `'DM-C'` : simple Difference in Means, cluster design
- `'DM-C($0.75$)'` : thresholded Difference in Means, cluster design
- `'PI-$n(B;1)$'` : polynomial interpolation estimator under simple Bernoulli staggered rollout design, assuming a degree $\beta=1$ model
- `'LS-Prop(1)'` : least squares estimator regressing over proportion of neighbors treated, Bernoulli design, assuming a degree $\beta=1$ model
- `'LS-Num(1)'` : least squares estimator regressing over number of neighbors treated, Bernoulli design, assuming a degree $\beta=1$ model
- `'PI-$n(B;2)$'` : polynomial interpolation estimator under simple Bernoulli staggered rollout design, assuming a degree $\beta=2$ model
- `'LS-Prop(2)'` : least squares estimator regressing over proportion of neighbors treated, Bernoulli design, assuming a degree $\beta=2$ model
- `'LS-Num(2)'` : least squares estimator regressing over number of neighbors treated, Bernoulli design, assuming a degree $\beta=2$ model
- `'PI-$n(B;3)$'` : polynomial interpolation estimator under simple Bernoulli staggered rollout design, assuming a degree $\beta=3$ model
- `'LS-Prop(3)'` : least squares estimator regressing over proportion of neighbors treated, Bernoulli design, assuming a degree $\beta=3$ model
- `'LS-Num(3)'` : least squares estimator regressing over number of neighbors treated, Bernoulli design, assuming a degree $\beta=3$ model
- `'DM'` : simple Difference in Means, Bernoulli design
- `'DM($0.75$)'` : thresholded Difference in Means, Bernoulli design

NOTE: When choosing estimators to plot, make sure they remain in the same relative order as the list above to ensure consistency of colors in the plots. 


## Increasing Edges 
This folder has the files related to the experiment varying the cluster quality with respect to edges crossing clusters.
- `main_increasing_edges.py` is the experiment (fix all parameters, but vary $p_\text{in}$ and $p_\text{out}$, keeping average degree fixed, and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `myFunctions_edges.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network; yes, essentially same file as the corresponding two with the same function in the other folders
- `plots.py` script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
- `plots` is a folder containing the resulting plots from `plots.py`(there is a seperate folder for each model degee)

### Running Increasing Edges Experiments
These experiments simply compare the performance of different estimators as the graph connectivity is varied. In particular, we vary the edge probabilities between units of the same cluster ($p_\text{in}$) and units of different clusters ($p_\text{out}$), from perfectly disjoint clusters ($p_\text{in} = 0.5, p_\text{out} = 0$) to Erd\"os R\'enyi ($p_\text{in} = 0.01, p_\text{out} = 0.01$). In all experiments, the average degree is kept constant at $10$.

### Plotting Results
TBD...