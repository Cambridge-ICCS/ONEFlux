import numpy as np
from scipy.optimize import curve_fit
from oneflux_steps.ustar_cp_python.fcNaniqr import fcNaniqr
from oneflux_steps.ustar_cp_python.fcEqnAnnualSine import fcEqnAnnualSine
from oneflux_steps.ustar_cp_python.fcr2Calc import fcr2Calc
from typing import Tuple
from numpy.typing import NDArray
from oneflux_steps.ustar_cp_python.fcBin import fcBin

def cpdAssignUStarTh20100901(*args):
    return None, None, None, None, None, None, None, None, None, None, None, None


def identifyOutliers(x_norm_x, threshold):
    """
    Identifies outliers based on standardized scores.

    Parameters:
    x_norm_x (numpy.ndarray): Array of normalized values.
    threshold (float): Threshold value for outlier detection.

    Returns:
    tuple: A boolean array (f_out) indicating outliers and an array (i_out) of outlier indices.
    """
    f_out = (x_norm_x > threshold)
    i_out = np.where(f_out)[0]  # Indices of outliers

    return f_out, i_out

def computeStandardizedScores(x):
    """Standardizes the matrix x by its median and interquartile range."""
    mx = np.nanmedian(x, axis=0)  # Compute median ignoring NaNs
    iqr = fcNaniqr(x)
    x_norm = (x - mx) / iqr  # Standardize

    return np.nanmax(np.abs(x_norm), axis=1, keepdims=True)  # Max absolute standardized score per row



def fitAnnualSineCurve(mt, Cp, iSelect):
    """
    Fits an annual sine curve to the selected data points and returns
    the sine coefficients and R-squared value.

    Parameters
    ----------
    mt : array-like
        The time data (e.g., day of year).
    Cp : array-like
        The corresponding values to be fitted.
    iSelect : array-like of bool
        A boolean mask or indices that indicate which points to use
        for the fitting.

    Returns
    -------
    sSine : np.ndarray
        A 1D array of length 4:
          [offset, amplitude, phase, rSquared]
    """
    # Prepare the data
    xdata = np.array(mt)[iSelect]
    ydata = np.array(Cp)[iSelect]

    # Define a local function for curve_fit: curve_fit expects
    # a callable f(t, b0, b1, b2) with first arg = x, subsequent = params
    def _annual_sine_for_curve_fit(t, b0, b1, b2):
        return fcEqnAnnualSine(np.asarray([b0, b1, b2]), t)

    # Initial guess for [offset, amplitude, phase]
    initial_guess = [1.0, 1.0, 1.0]

    # Perform the fit via non-linear regression
    popt, _ = curve_fit(_annual_sine_for_curve_fit, xdata, ydata, p0=initial_guess)

    # Compute predicted values for the fitted parameters
    predictedCp = fcEqnAnnualSine(np.asarray(popt), xdata)

    # Compute R-squared
    r2 = fcr2Calc(ydata, predictedCp)

    # Adjust phase to be in the range [0, 365.25)
    popt[2] = popt[2] % 365.25

    # Return fitted coefficients plus the R-squared value
    # List containing ndarray to make output format match matlab for comparative testing
    sSine = np.array([popt[0], popt[1], popt[2], r2], dtype=float)

    return sSine

def aggregateSeasonalAndAnnualValues(
    xCp, 
    iSelect, 
    nDim, 
    nWindows, 
    nStrata, 
    nBoot
):
    """
    Python equivalent of the MATLAB function aggregateSeasonalAndAnnualValues.

    Parameters
    ----------
    xCp : array-like
        Can be 2D [nWindows, nBoot] or 3D [nWindows, nStrata, nBoot].
    iSelect : array-like of int
        1-based linear indices in MATLAB's column-major ordering that should be selected.
    nDim : int
        2 or 3 (to indicate if xCp is 2D or 3D).
    nWindows : int
    nStrata : int
    nBoot : int

    Returns
    -------
    CpA : np.ndarray
        Aggregated mean of selected change points along the appropriate dimension.
        For nDim=2, shape is (nBoot,). For nDim=3, shape is (nBoot,).
    nA : np.ndarray
        Count of non-NaN selected points. Same shape as CpA.
    xCpSelect : np.ndarray
        Same shape as xCp, with NaN everywhere except the selected positions.
    """
    # Prepare an output array (same shape) filled with NaNs
    xCpSelect = np.full_like(xCp, np.nan, dtype=float)

    # Treat selection arrays an array of integers
    # (this allows both an array of booleans and an array of integers to be used
    # for iselect)
    iSelect_array = np.asarray(iSelect, dtype=int)

    # Aggregate values based on dimensions
    if nDim == 2:
        # mask xCp using iSelect_array
        xCpSelect = xCp.copy()
        for i in range(len(xCp)):
            if iSelect_array[i] == 0:
                xCpSelect[i] = np.nan

        xCpGF = xCpSelect  # naming convention

        # xCp shape = [nWindows, nBoot]
        CpA = np.nanmean(xCpGF)
        nA  = np.sum(~np.isnan(xCpSelect))
    elif nDim == 3:
        
        # mask xCp using iSelect_array
        xCpSelect = xCp.copy()
        for i in range(len(xCp)):
            for j in range(len(xCp[i])):
              if iSelect_array[i][j] == 0:
                  xCpSelect[i][j] = np.nan

        xCpGF = xCpSelect  # Naming convention

        # xCp shape = [nWindows, nStrata, nBoot]
        # reshape => (nWindows*nStrata, nBoot) in column-major
        xCpGF_reshaped = np.reshape(xCpGF, (nWindows * nStrata, nBoot), order='F')
        CpA = np.nanmean(xCpGF_reshaped, axis=0)
        nA  = np.sum(~np.isnan(xCpGF_reshaped), axis=0)
    else:
        raise ValueError("Invalid number of dimensions: Expected 2D or 3D Stats.")

    return CpA, nA, xCpSelect


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

def updateSelectedIndices(iSelect : np.ndarray, iOut : np.ndarray, fSelect : np.ndarray, fOut : np.ndarray) -> (np.ndarray, int, np.ndarray):
    """
    Parameters
    ----------
    iSelect : np.ndarray (int)
        Array of selected indices.
    iOut : np.ndarray (int)
        Array of outlier indices to remove from iSelect.
    fSelect : np.ndarray (bool)
        Boolean array indicating whether each point is selected.
    fOut : np.ndarray (bool)
        Boolean array indicating outlier points.

    Returns
    -------
    iSelect : np.ndarray (int)
        Updated array of selected indices after outlier removal.
    nSelect : int
        Length of the updated iSelect.
    fSelect : np.ndarray (bool)
        Updated boolean array with outliers removed from selection.
    """
    # Remove outlier indices from iSelect
    iSelect = np.setdiff1d(iSelect, iOut)

    # Count the remaining selected indices
    nSelect = len(iSelect)

    # Exclude outliers from fSelect (element-wise "and not")
    fSelect = fSelect & np.logical_not(fOut)

    return iSelect, nSelect, fSelect
