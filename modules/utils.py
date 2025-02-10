import numpy as np

import matplotlib as mpl

def set_plot_configs():
    # Define matplotlib style
    mpl.style.use('classic')
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

    print("  Shape:",array.shape)
    if np.any(np.isnan(array)):
        print("  NaNs: ",np.sum(np.isnan(array)))
    print("  Min:  ",np.nanmin(array))
    print("  Max:  ",np.nanmax(array))
    print("  Mean: ",np.nanmean(array))
    print("  Std:  ",np.nanstd(array))