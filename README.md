The folder **Covariate Community** Correlation has the files related to the experiment varying $\phi$ aka homophily, aka cluster quality with respect to covariate balance.
- `main_correlation.py` is the experiment (fix all parameters, but vary $\phi$ and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `experiment_functions.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network
- `plots.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
- `plots` is a folder containing the resulting plots from `plots.py` (there is a seperate folder for each model degee)

The folder **Increasing Edges** has the files related to the experiment varying the cluster quality with respect to edges crossing clusters.
- `main_increasing_edges.py` is the experiment (fix all parameters, but vary $p_\text{in}$ and $p_\text{out}$, keeping average degree fixed, and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `myFunctions_edges.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network--note this is essentially the same file as `experiment_functions.py` above
- `plots.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
- `plots` is a folder containing the resulting plots from `plots.py` (there is a seperate folder for each model degee)

The folder **Robustness to Misspecification** has the files related to the experiments that compare the performance of different estimators and designs under model misspecification.
- `main_misspecification.py` is the experiment (fix all parameters except $\phi$ and $p$, and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `experiment_functions.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network--note this is essentially the same file as `experiment_functions.py` above
- `plots_rm.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each true model
    - for example, the subfolder `ppom1` means that the results in that folder correspond to an experiment where the true model was the polynomial potential outcomes model of degree $\beta=1$)
- `plots` is a folder containing the resulting plots from `plots.py` (there is a seperate folder for each true model)
