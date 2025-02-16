from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from oneflux_steps.ustar_cp_python.fcBin import fcBin

def aggregateSeasonalMeans(mt: NDArray, Cp: NDArray, xmt: NDArray, iSelect: NDArray, nWindows: int, nStrata: int, nBoot: int)-> Tuple[NDArray, NDArray]:
    """
    Python equivalent of the MATLAB function aggregateSeasonalMeans.
    Aggregates seasonal means for time and change points.

    Parameters
    ----------
    mt : array-like
        The time data.
    Cp : array-like
        The corresponding change point values.
    xmt : array-like
        A reference array that must reshape cleanly to (nWindows, nStrata * nBoot).
        Used to compute the median number of windows (nW).
    iSelect : array-like of int
        Numeric indices (1-based in MATLAB) of selected measurements to use.
        For Python (0-based), ensure you pass valid 0-based indices.
    nWindows : int
    nStrata : int
    nBoot : int

    Returns
    -------
    tW : np.ndarray
        The seasonal mean times for each bin.
    CpW : np.ndarray
        The seasonal mean change points for each bin.
    """

    # ----------------------------------------------------------
    # 1) Calculate Median Number of Windows
    #    In MATLAB: 
    #       reshape(xmt, nWindows, nStrata * nBoot)
    #    Here we reshape the 1D array xmt => shape (nWindows, nStrata*nBoot)
    # ----------------------------------------------------------
    try:
        xmt_reshaped = xmt.reshape(nWindows, nStrata * nBoot)
    except ValueError:
        raise ValueError(
            f"Cannot reshape xmt of length {xmt.size} "
            f"to shape ({nWindows}, {nStrata*nBoot})."
        )

    # sum(~isnan(...)) in MATLAB => np.sum(~np.isnan(...)) in NumPy
    # nanmedian(...) => np.nanmedian(...)
    # => sum over axis=1 if you want row sums (assuming typical usage)
    # but in the MATLAB code it's sum( (nWindows, nStrata*nBoot) ) => axis=0 by default
    nW_array = np.sum(~np.isnan(xmt_reshaped), axis=0)  # shape: (nStrata*nBoot,)
    nW = np.nanmedian(nW_array)

    # ----------------------------------------------------------
    # 2) Sort Selected Measurements
    #    MATLAB:
    #      [mtSelect, i] = sort(mt(iSelect));
    #      CpSelect = Cp(iSelect(i));
    # ----------------------------------------------------------
    # *In Python*, iSelect is an array of *0-based* indices.
    # We then do:
    iSelect = iSelect.astype(int) 
    selected_mt = mt[iSelect]
    # Sort them while keeping track of the sorted order
    sort_order = np.argsort(selected_mt)  # array of int
    mtSelect = selected_mt[sort_order]
    # Now reorder Cp similarly
    selected_Cp = Cp[iSelect]
    CpSelect = selected_Cp[sort_order]

    # ----------------------------------------------------------
    # 3) Define Bins Based on Percentiles
    #    xBins = prctile(mtSelect, 0:(100/nW):100);
    # In Python, np.percentile(...)
    # and we create an array of percentiles from 0 to 100 in steps of (100/nW)
    # But nW might be float => we do something like
    # np.linspace(0, 100, int(nW)+1) or something similar. However,
    # to replicate MATLAB's 0:(100/nW):100 exactly, we must handle floats carefully.
    # We'll do:
    if nW < 1:
        # Fallback: if nW < 1 for some reason, do 1 bin, i.e. [0, 100]
        pct_array = [0, 100]
    else:
        step = 100.0 / nW  # step size
        # The number of steps is int(nW)+1 if nW is integer-like. 
        # However, if nW is float, MATLAB still enumerates 0, step, 2*step,..., up to 100.
        # We'll replicate that logic using np.arange, then clip at <=100
        pct_array = []
        current = 0.0
        while current < 100.0 + 1e-9:
            pct_array.append(current)
            current += step
        # In case floating arithmetic overshoots 100 a bit
        pct_array[-1] = 100.0

    # Now compute bin edges from these percentiles
    xBins = np.percentile(mtSelect, pct_array)

    # ----------------------------------------------------------
    # 4) fcBin equivalent in Python
    #    [~, tW, CpW] = fcBin(mtSelect, CpSelect, xBins, 0);


    # Run fcBin
    nCount, tW, CpW = fcBin(mtSelect, CpSelect, xBins, 0)

    return tW, CpW
