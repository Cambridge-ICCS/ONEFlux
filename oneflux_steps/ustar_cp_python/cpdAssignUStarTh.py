import numpy as np
from scipy.optimize import curve_fit
from oneflux_steps.ustar_cp_python.fcNaniqr import fcNaniqr
from oneflux_steps.ustar_cp_python.fcEqnAnnualSine import fcEqnAnnualSine
from oneflux_steps.ustar_cp_python.fcr2Calc import fcr2Calc




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