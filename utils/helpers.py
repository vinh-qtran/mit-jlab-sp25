import numpy as np

def check_array(array,log=False):
    if log:
        array = np.log10(array)

    print("  Shape:",array.shape)
    if np.any(np.isnan(array)):
        print("  NaNs: ",np.sum(np.isnan(array)))
    print("  Min:  ",np.nanmin(array))
    print("  Max:  ",np.nanmax(array))
    print("  Mean: ",np.nanmean(array))
    print("  Std:  ",np.nanstd(array))