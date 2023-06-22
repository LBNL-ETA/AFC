# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Plotting module.
"""

# pylint: disable=invalid-name, too-many-arguments

import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
# import matplotlib.dates as mdates
# from matplotlib.offsetbox import AnchoredText

def plot_plot_xy(axs, df, ylim=None, legend_loc=2, title=None, ylab=None, labels=None):
    """
        Function to plot emulator results in a multiple plots.
        Input
        -----
            axs (matplotlib.axes._subplots.AxesSubplot): A matlibplot object for the axes.
            df (pandas df): The pandas dataframe to be plotted.
            ylim (array[2] or None): The limits for the y-axis as array of lower [0]
                                     and upper [1] limit. (default=None)
            legend_loc (int or None): The location of the ledgend. (default=2)
            title (str or None): The title of the plot. (default=None)
            ylab (str or None): The y-axis label of the plot. (default=None)
            labels (array or None): The labels for the series. (default=None)
            
        Returns
        -------
            none
    """
    axs.plot(df)
    if ylim:
        axs.set_ylim(ylim)
    if legend_loc:
        axs.legend(df.columns, loc=legend_loc)
    if title:
        axs.set_title(title)
    if ylab:
        axs.set_ylabel(ylab)
    if labels:
        axs.legend(labels, loc=legend_loc)

def plot_standard1(data, title=None, plot=True, tight=True):
    """
        Function to plot emulator results in a standard plot template.
        Input
        -----
            data (pandas df): The pandas dataframe as result from the emulator.
            title (str or none): The title of the plot, or None. (default=None)
            plot (bool): Flag to plot the plot or return references. (default=True)
            tight (bool): Flag to enable tight_layout. (Defualt=True)

        Returns
        -------
            [fig, axs] or none: Returns the figure and axis objects if plot=False, otherwise none.
    """
    n = 8
    fig, axs = plt.subplots(n,1, figsize=(12, n*3), sharex=True, sharey=False)
    if title:
        fig.suptitle(title, fontsize=16)
    axs = axs.ravel()
    plot_plot_xy(axs[0],data[['GHI','DHI','DNI']],title='Weather',ylab='Solar Irradiance [W/m2]',
                 labels=['Global Horizontal','Diffuse Horizontal','Direct Normal',])
    plot_plot_xy(axs[1],data[[c for c in data.columns if 'Facade State' in c]],
                 title='Facade State',ylab='State [1]',
                 labels=['Shade bottom','Shade middle','Shade top'])
    plot_plot_xy(axs[2],data[['Work Plane Illuminance [lx]']],
                 title='Work Plane Illuminance',ylab='Illuminance [lx]',
                 labels=['Workplane Illuminance'])
    axs[2].plot(data[['Work Plane Illuminance Min [lx]']], color='black', linestyle='--')
    tt = data[['Solar Heat Gain [W]','Power Plugs [W]','Power Occupancy [W]','Power Lights [W]']]
    plot_plot_xy(axs[3], tt, title='Thermal Load', ylab='Thermal Power [W]',
                 labels=['Solar Heat Gain','Occupancy','Plugload Power','Lighting Power'])
    tt = data[['Power Cooling [W]','Power Heating [W]','Power Lights [W]','Power Plugs [W]']]
    plot_plot_xy(axs[4], tt, title='Electric Load', ylab='Electric Power [W]',
                 labels=['Cooling Power','Heating Power','Lighting Power','Plugload Power'])
    tt = data[['Temperature 0 [C]','Temperature 1 [C]']]
    plot_plot_xy(axs[5], tt, title='Temperature', ylab='Temperature [C]',
                 labels=['Room','Slab'])
    axs[5].plot(data[['Temperature 0 Max [C]']], color='blue', linestyle='--')
    axs[5].plot(data[['Temperature 0 Min [C]']], color='red', linestyle='--')
    axs[5].plot(data[['Temperature 1 Max [C]']], color='blue', linestyle=':')
    axs[5].plot(data[['Temperature 1 Min [C]']], color='red', linestyle=':')
    plot_plot_xy(axs[6],data[['Glare [-]']],title='Glare Level',ylab='DGP [1]',
                 labels=['Glare'])
    axs[6].plot(data[['Glare Max [-]']], color='black', linestyle='--')
    plot_plot_xy(axs[7],data[['Import Power [kW]']],
                 title='Electric Power',ylab='Electric Power [W]',
                 labels=['Electric Power (Cool+Heat+Light+Plug)'])
    if tight:
        plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.85)
    if plot:
        plt.show()
        return None
    return fig, axs
