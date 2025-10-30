
# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Reduced-order RC tuning utility.
"""

# pylint: disable=too-many-arguments, bare-except, too-many-positional-arguments

import matplotlib.pyplot as plt

def convert_rc_parameter(model, print_new=False):
    '''convert rc parameters to dict'''
    param_new = {}
    for p in model.rc_parameter:
        param_new[p] = getattr(model, p).value

    if print_new:
        for k,v in convert_rc_parameter(model).items():
            print(k+'\t', v)
    return param_new

def plot_streams(axs, temp, title=None, ylabel=None, legend=False, loc=1):
    '''
        Utility to simplify plotting of subplots.

        Input
        -----
            axs (matplotlib.axes._subplots.AxesSubplot): The axis to be plotted.
            temp (pandas.Series): The stream to be plotted.
            plot_times (bool): Flag if time separation should be plotted. (default=True)
    '''
    axs.plot(temp)
    axs.legend(temp.columns, loc=1)
    if title:
        axs.set_title(title)
    if ylabel:
        axs.set_ylabel(ylabel)
    if legend:
        axs.legend(legend, loc=loc)

def plot_standard1(df, plot=True, tight=True):
    '''
        A standard plotting template to present results.

        Input
        -----
            df (pandas.DataFrame): The resulting dataframe with the optimization result.
            plot (bool): Flag to plot or return the figure. (default=True)
            plot_times (bool): Flag if time separation should be plotted. (default=True)
            tight (bool): Flag to use tight_layout. (default=True)
            
        Returns
        -------
            None if plot == True.
            else:
                fig (matplotlib figure): Figure of the plot.
                axs (numpy.ndarray of matplotlib.axes._subplots.AxesSubplot): Axis of the plot.
    '''
    n = 6
    fig, axs = plt.subplots(n,1, figsize=(12, n*3), sharex=True, sharey=False,
                            gridspec_kw={'width_ratios':[1]})
    axs = axs.ravel()
    plot_streams(axs[0], df[['Temperature 0 [C]','Measured Room Temperature [C]']])
    plot_streams(axs[1], df[['Temperature 1 [C]','Measured Slab Temperature [C]']])
    if 'Temperature 2 [C]' in df.columns:
        plot_streams(axs[2], df[['Temperature 2 [C]','Measured Wall Temperature [C]']])
    plot_streams(axs[3], df[['Convective Internal Gains [W]','Radiative Internal Gains [W]']])
    plot_streams(axs[4], df[['Window Absorption 1 [W]','Window Absorption 2 [W]']])
    plot_streams(axs[5], df[['Outside Air Temperature [C]']])
    #plot_streams(axs[6], df[['Window Heat Transfer Coefficient [W/K]']])
    if plot:
        if tight:
            plt.tight_layout()
        plt.show()
        return None
    return fig, axs
