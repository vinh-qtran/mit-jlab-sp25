import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def set_plot_configs():
    # Define matplotlib style
    plt.style.use('classic')
    mpl.rc('xtick', labelsize=23); mpl.rc('ytick', labelsize=23)
    mpl.rc('xtick.major', size=15 , width=2)
    mpl.rc('xtick.minor', size=8, width=2, visible=True)
    mpl.rc('ytick.major', size=15 , width=2)
    mpl.rc('ytick.minor', size=8, width=2, visible=True)
    mpl.rc('lines',linewidth=3, markersize=20)
    mpl.rc('axes', linewidth=2, labelsize=30, labelpad=2.5)
    mpl.rc('legend', fontsize=25, loc='best', frameon=False, numpoints=1)

    mpl.rc('font', family='STIXGeneral')
    mpl.rc('mathtext', fontset='stix')
    

def check_array(array,log=False):
    # Check the shape and statistics of an array
    if log:
        array = np.log10(array)

    print(" Shape:",array.shape)
    if np.any(np.isnan(array)):
        print(" NaNs: ", f'{np.sum(np.isnan(array))}')
    print(" Min:  ", f'{np.nanmin(array):.2e}')
    print(" Max:  ", f'{np.nanmax(array):.2e}')
    print(" Mean: ", f'{np.nanmean(array):.2e}')
    print(" Std:  ", f'{np.nanstd(array):.2e}')