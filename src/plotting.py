import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_series(features_names, id, dataframe, axes, ox, oy, stim_start, stim_end, wrong_bool=None):

    # Calculating minimum value
    min_val = dataframe[features_names[0]][dataframe.id==id].values[0]
    for feat_name in features_names:
        min_val_aux = min(dataframe[feat_name][dataframe.id==id])
        min_val = min(min_val, min_val_aux)

    # Calculating maximum value
    max_val = dataframe[features_names[0]][dataframe.id==id].values[0]
    for feat_name in features_names:
        max_val_aux = max(dataframe[feat_name][dataframe.id==id])
        max_val = max(max_val, max_val_aux)

    # Plotting features
    for feat_name in features_names:
        axes[ox,oy].plot(dataframe.time[dataframe.id==id], dataframe[feat_name][dataframe.id==id], linewidth = '2.0')
    
    # Plotting start and end of stimulus
    axes[ox,oy].plot([stim_start, stim_start], [min_val, max_val])
    axes[ox,oy].plot([stim_end, stim_end], [min_val, max_val])

    # Plotting misclassified examples
    if wrong_bool is not None:
        for feat_name in features_names:
            axes[ox,oy].scatter(dataframe.time[dataframe.id==id][wrong_bool], dataframe[feat_name][dataframe.id==id][wrong_bool], label=feat_name+' mis')

    # Axis labels and legend
    ylabel_str = features_names[0]
    for i in range(1, len(features_names)):
        ylabel_str += ' and '+features_names[i]
    
    xlabel_str = 'Time'
    axes[ox,oy].set_title(ylabel_str + ' VS ' + xlabel_str)
    axes[ox,oy].set_xlabel(xlabel_str)
    axes[ox,oy].set_ylabel(ylabel_str)
    axes[ox,oy].legend()




def plot_allSeries_byID(id, dataframe, wrong_bool=None):
    # nrows values is set according to the project
    nrows = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(17)
    fig.subplots_adjust(hspace=.5)

    stim_start = dataframe.t0[dataframe.id==id].values[0]
    stim_end = dataframe.t0[dataframe.id==id].values[0] + dataframe.dt[dataframe.id==id].values[0]

    plot_series(['Humidity'], id, dataframe, axes, 0, 0, stim_start, stim_end, wrong_bool)
    plot_series(['Temp.'], id, dataframe, axes, 0, 1, stim_start, stim_end, wrong_bool)
    plot_series(['R1', 'R2'], id, dataframe, axes, 1, 0, stim_start, stim_end, wrong_bool)
    plot_series(['R3', 'R4'], id, dataframe, axes, 1, 1, stim_start, stim_end, wrong_bool)
    plot_series(['R5', 'R6'], id, dataframe, axes, 2, 0, stim_start, stim_end, wrong_bool)
    plot_series(['R7', 'R8'], id, dataframe, axes, 2, 1, stim_start, stim_end, wrong_bool)


    stims = sorted(list(set(dataframe['class'][dataframe['id'] == id])))
    stim = stims[-1]
    plt.suptitle('Sensor Reading on Day '+str(id)+' ('+stim+')')




def plot_misclassified_byID(df_test, id, y_true, y_pred):
    unique_ids = list(set(df_test.id))

    # Checking introduced ID is in test set
    if id not in unique_ids:
        raise ValueError('series ID not in test set. Check introduced ID')

    # Selecting samples by ID
    df_aux = df_test[df_test.id==id]

    # Selecting misclassified samples (by hand, we have no ID info at y_pred & y_true)
    nsamples = 0
    idx = 0
    while unique_ids[idx] != id:
        nsamples += len(df_test[df_test['id']==unique_ids[idx]])
        idx +=1

    wrong_bool = (y_pred != y_true)
    wrong_bool = wrong_bool[nsamples:nsamples+len(df_aux)]

    plot_allSeries_byID(id, df_test, wrong_bool=wrong_bool)

