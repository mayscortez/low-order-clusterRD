import seaborn.objects as so
import pandas as pd
import pickle
import sys

# basic plot:            col = nc, row = beta
# compare clustering:    col = clustering, row = nc

def main():
    args = sys.argv[1:]
    input_file = args[0]    # location of experiment data pkl file
    output_file = args[1]   # location to save plot image

    col_var = None
    row_var = None
    if len(args) > 2: col_var = args[2]   # variable facet plots by column
    if len(args) > 3: row_var = args[3]   # variable to facet plots by row

    data_file = open(input_file, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    df = pd.DataFrame(data)

    colors = ['#0296fb', '#e20287']

    p = (
        so.Plot(df, x='q', y='tte_hat', color='est')
        .facet(row=row_var, col=col_var)
        .scale(color=colors)
        .add(so.Line(), so.Agg())                 # line plot of expected bias
        .add(so.Band(), so.Est(errorbar='sd'))    # shading for standard deviation
        .layout(size=(10,8))
    )

    p.show()
    p.save(output_file)

if __name__ == '__main__':
    main()