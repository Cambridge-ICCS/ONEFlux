import numpy as np
from scipy.optimize import curve_fit
from oneflux_steps.ustar_cp_python.fcReadFields import fcReadFields
from oneflux_steps.ustar_cp_python.fcNaniqr import fcNaniqr
from oneflux_steps.ustar_cp_python.fcEqnAnnualSine import fcEqnAnnualSine
from oneflux_steps.ustar_cp_python.fcr2Calc import fcr2Calc
from oneflux_steps.ustar_cp_python.fcx2colvec import fcx2colvec
from typing import Tuple
import json
from numpy.typing import NDArray
from oneflux_steps.ustar_cp_python.fcBin import fcBin
from oneflux_steps.ustar_cp_python.utilities import index_or_mark_array_update, nlinfit


def cpdAssignUStarTh20100901(Stats, plotFlag, siteYearText, *args):
    """
    Parameters:
    -----------
    Stats : array-like or str
        Stats structure that may be JSON-encoded (depending on *args).
    plotFlag : bool or int
        Flag indicating whether to plot (unused in this excerpt).
    siteYearText : str
        String describing site year (unused in this excerpt).
    *args : list
        Variable-length argument list. Can contain instructions like
        ['jsondecode', 1], specifying that Stats should be decoded from JSON.

    Returns:
    --------
    annualChangePoint : np.ndarray
    numAnnualSelected : np.ndarray
    seasonalTimeWindow : np.ndarray
    seasonalChangePoint : np.ndarray
    dominantMode : str
    failureMessage : str
    selectedPointsFlag : np.ndarray (bool)
    sineCurve : np.ndarray
    fractionSignificant : float
    fractionModeD : float
    fractionSelected : float

    The logic follows the MATLAB code step-by-step, including:
    - Decoding JSON if required.
    - Checking dimensions of Stats.
    - Extracting relevant fields (like Cp, b1, p).
    - Determining significance and modes (D/E).
    - Excluding outliers, aggregating results.
    - Fitting annual sine curve.
    """

    # -------------------------------------------------------------------------
    # 1) Initialize Output Variables

    annualChangePoint = []
    numAnnualSelected = []
    seasonalTimeWindow = []
    seasonalChangePoint = []
    selectedPointsFlag = []
    dominantMode = ""
    failureMessage = ""
    sineCurve = []
    fractionSignificant = []
    fractionModeD = []
    fractionSelected = []

    # -------------------------------------------------------------------------
    # 2) Decode JSON if Required
    # TODO - PROBABLY REMOVE

    # In MATLAB, the code checks 'varargin' for a cell array with 'jsondecode'.
    # In Python, we check if `args` includes something like ['jsondecode', 1].
    for arg in args:
        # Example check: if arg is a list, arg[0] might be 'jsondecode'
        if isinstance(arg, list) and len(arg) > 0 and arg[0] == "jsondecode":
            # Then check subsequent entries for numeric IDs
            for j in arg[1:]:
                # If j == 1, decode Stats from JSON
                if j == 1:
                    if isinstance(Stats, str):
                        Stats = json.loads(Stats)

    # -------------------------------------------------------------------------
    # 3) Determine Dimension Sizes of Stats

    # The MATLAB code checks ndims(Stats) and shape(Stats).
    # We assume Stats is a NumPy array (or a nested list that's been
    # converted to an array).
    if not isinstance(Stats, np.ndarray):
        # Attempt to convert Stats to a NumPy array if it's still a list
        Stats = np.array(Stats, dtype=object)

    numDimensions = Stats.ndim

    if numDimensions == 2:
        numWindows, numBootstraps = Stats.shape
        numTemperatureStrata = 1
        temperatureStrataFactor = 0.5

    elif numDimensions == 3:
        numWindows, numTemperatureStrata, numBootstraps = Stats.shape
        temperatureStrataFactor = 1

    else:
        failureMessage = "Stats must be 2D or 3D."
        return (
            annualChangePoint,
            numAnnualSelected,
            seasonalTimeWindow,
            seasonalChangePoint,
            dominantMode,
            failureMessage,
            selectedPointsFlag,
            sineCurve,
            fractionSignificant,
            fractionModeD,
            fractionSelected,
        )

    # -------------------------------------------------------------------------
    # 4) Set Reference Values

    referenceWindows = 4
    requiredSelectionCount = referenceWindows * temperatureStrataFactor * numBootstraps

    # -------------------------------------------------------------------------
    # 5) Preallocate Outputs

    annualChangePoint = np.full((numBootstraps,), np.nan)
    numAnnualSelected = np.full((numBootstraps,), np.nan)
    seasonalTimeWindow = np.full((numWindows,), np.nan)
    seasonalChangePoint = np.full((numWindows,), np.nan)

    # -------------------------------------------------------------------------
    # 6) Extract Variables from Stats Structure
    # The MATLAB code uses fcReadFields and fcx2colvec to read each field and reshape.
    # Here, we assume Python equivalents: readFields(Stats, varName) and x2colvec().
    # In an actual implementation, these must be defined or replaced by direct indexing.

    variableNames = ["mt", "Cp", "b1", "c2", "cib1", "cic2", "p"]
    # We'll store them in a dictionary by field name.
    b1 = fcx2colvec(fcReadFields(Stats, "b1"))
    c2 = fcx2colvec(fcReadFields(Stats, "c2"))
    cib1 = fcx2colvec(fcReadFields(Stats, "cib1"))
    cic2 = fcx2colvec(fcReadFields(Stats, "cic2"))
    p = fcx2colvec(fcReadFields(Stats, "p"))

    measurementTime = fcReadFields(Stats, "mt")
    mt = fcx2colvec(measurementTime)
    changePoint = fcReadFields(Stats, "Cp")
    Cp = fcx2colvec(changePoint)

    # -------------------------------------------------------------------------
    # 7) Identify Significant Change Points

    significanceThreshold = 0.05
    # p <= threshold is True/False mask
    significantFlag = p <= significanceThreshold

    # -------------------------------------------------------------------------
    # 8) Identify Model Type (2-parameter vs 3-parameter)

    # If all c2 are NaN => effectively 2 parameters
    # The MATLAB code checks sum(~isnan(c2)) == 0
    if np.sum(~np.isnan(c2)) == 0:
        numParameters = 2
        c2 = np.zeros_like(b1)
        cic2 = np.zeros_like(b1)
    else:
        numParameters = 3

    # -------------------------------------------------------------------------
    # 9) Classify Significant Change Points:

    # Indices that are valid (not NaN in b1+c2+Cp):
    validMask = ~np.isnan(b1 + c2 + Cp)
    validIndices = np.where(~np.isnan(mt))[0]
    numValidMeasurements = len(validIndices)

    # Non-significant
    nonSignificantIndices = np.where((~significantFlag) & validMask)[0]
    numNonSignificant = len(nonSignificantIndices)

    # Significant
    significantIndices = np.where(significantFlag & validMask)[0]
    numSignificant = len(significantIndices)

    # Mode E: b1 < c2
    modeEIndices = np.where(significantFlag & (b1 < c2) & validMask)[0]
    numModeE = len(modeEIndices)

    # Mode D: b1 >= c2
    modeDIndices = np.where(significantFlag & (b1 >= c2) & validMask)[0]
    numModeD = len(modeDIndices)

    # Decide dominant mode
    if numModeD >= numModeE:
        selectedIndices = modeDIndices
        dominantMode = "D"
    else:
        selectedIndices = modeEIndices
        dominantMode = "E"

    numSelected = len(selectedIndices)

    # Update Selection Flags
    selectedPointsFlag = np.zeros_like(significantFlag, dtype=bool)
    selectedPointsFlag[selectedIndices] = True

    modeDFlag = np.full_like(significantFlag, np.nan, dtype=float)
    modeDFlag[modeDIndices] = 1

    modeEFlag = np.full_like(significantFlag, np.nan, dtype=float)
    modeEFlag[modeEIndices] = 1

    # Fractions
    if numValidMeasurements > 0:
        fractionSignificant = numSignificant / numValidMeasurements
        fractionModeD = numModeD / max(numSignificant, 1)  # avoid zero-div
        fractionSelected = numSelected / numValidMeasurements
    else:
        fractionSignificant, fractionModeD, fractionSelected = (0, 0, 0)

    # -------------------------------------------------------------------------
    # 10) Abort if Too Few Selections
    # This first if clause is to replciate the case where 0 divided by zero evaluates to Nan and farctionSlected is never evaluatedfractionSelected
    if numSelected == 0 and numValidMeasurements == 0:
        pass
    elif fractionSelected < 0.10:
        failureMessage = "Less than 10% successful detections."
        return (
            annualChangePoint,
            numAnnualSelected,
            seasonalTimeWindow,
            seasonalChangePoint,
            dominantMode,
            failureMessage,
            selectedPointsFlag,
            np.array([]),
            fractionSignificant,
            fractionModeD,
            fractionSelected,
        )

    # -------------------------------------------------------------------------
    # 11) Configure Regression Matrix

    if numParameters == 2:
        regressionMatrix = np.column_stack([Cp, b1, cib1])
    else:
        regressionMatrix = np.column_stack([Cp, b1, c2, cib1, cic2])

    # -------------------------------------------------------------------------
    # 12) Exclude Outliers Based on Standardized Scores

    standardizedScores = computeStandardizedScores(regressionMatrix)  # user-defined
    outlierFlag, outlierIndices = identifyOutliers(
        standardizedScores, 5
    )  # user-defined

    selectedIndices, numSelected, selectedPointsFlag = updateSelectedIndices(
        selectedIndices, outlierIndices, selectedPointsFlag, outlierFlag
    )

    modeDIndices, numModeD = updateModes(modeDFlag, outlierIndices)  # user-defined
    modeEIndices, _ = updateModes(modeEFlag, outlierIndices)  # user-defined

    # Recompute significantIndices, etc.
    significantIndices = np.union1d(modeDIndices, modeEIndices)
    numSignificant = len(significantIndices)

    if numValidMeasurements > 0:
        fractionSignificant = numSignificant / numValidMeasurements
        fractionModeD = numModeD / max(numSignificant, 1)
        fractionSelected = numSelected / numValidMeasurements

    # -------------------------------------------------------------------------
    # 13) Check If Enough Change Points Remain

    if numSelected < requiredSelectionCount:
        failureMessage = (
            f"Too few selected change points: {numSelected}/{requiredSelectionCount}"
        )
        return (
            annualChangePoint,
            numAnnualSelected,
            seasonalTimeWindow,
            seasonalChangePoint,
            dominantMode,
            failureMessage,
            selectedPointsFlag,
            np.array([]),
            fractionSignificant,
            fractionModeD,
            fractionSelected,
        )

    # -------------------------------------------------------------------------
    # 14) Aggregate Seasonal and Annual Values

    annualChangePoint, numAnnualSelected, _ = aggregateSeasonalAndAnnualValues(
        changePoint,
        selectedIndices,
        numDimensions,
        numWindows,
        numTemperatureStrata,
        numBootstraps,
    )

    # -------------------------------------------------------------------------
    # 15) Aggregate Seasonal Means

    seasonalTimeWindow, seasonalChangePoint = aggregateSeasonalMeans(
        mt,
        Cp,
        measurementTime,
        selectedIndices,
        numWindows,
        numTemperatureStrata,
        numBootstraps,
    )

    # -------------------------------------------------------------------------
    # 16) Fit Annual Sine Curve

    sineCurve = fitAnnualSineCurve(mt, Cp, selectedIndices)

    # -------------------------------------------------------------------------
    # Return All Outputs in the Same Order as the MATLAB Code

    return (
        annualChangePoint,
        numAnnualSelected,
        seasonalTimeWindow,
        seasonalChangePoint,
        dominantMode,
        failureMessage,
        selectedPointsFlag,
        sineCurve,
        fractionSignificant,
        fractionModeD,
        fractionSelected,
    )


def identifyOutliers(x_norm_x, threshold):
    """
    Identifies outliers based on standardized scores.

    Parameters:
    x_norm_x (np.ndarray): Array of normalized values.
    threshold (float): Threshold value for outlier detection.

    Returns:
    tuple: A boolean array (f_out) indicating outliers and an array (i_out) of outlier indices.
    """
    f_out = x_norm_x > threshold
    i_out = np.where(f_out)[0]  # Indices of outliers

    return f_out, i_out


def computeStandardizedScores(x):
    """
    Compute standardized scores based on the median and interquartile range.

    Standardizes each column in the input matrix by subtracting the column median
    (ignoring NaNs) and dividing by the interquartile range (IQR). After standardizing,
    returns the maximum absolute standardized score for each row.

    Args:
        x (np.ndarray): Input data matrix of shape (n, m).

    Returns:
        numpy.ndarray: A (n, 1) column vector containing the maximum absolute
        standardised score for each row."""

    mx = np.nanmedian(x, axis=0)  # Compute median ignoring NaNs
    iqr = fcNaniqr(x)
    x_norm = (x - mx) / iqr  # Standardize

    return np.nanmax(
        np.abs(x_norm), axis=1, keepdims=True
    )  # Max absolute standardized score per row


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

    # Initial guess for [offset, amplitude, phase]
    initial_guess = [1.0, 1.0, 1.0]

    # Perform the fit via non-linear regression
    popt = nlinfit(xdata, ydata, fcEqnAnnualSine, initial_guess)

    # Compute predicted values for the fitted parameters
    predictedCp = fcEqnAnnualSine(np.asarray(popt), xdata)

    # Compute R-squared
    r2 = fcr2Calc(ydata, predictedCp)

    # Adjust phase to be in the range [0, 365.25)
    popt[2] = popt[2] % 365.25

    # Return fitted coefficients plus the R-squared value
    # List containing ndarray to make output format match matlab for comparative testing
    sSine = np.array([[popt[0], popt[1], popt[2], r2]], dtype=float)
    return sSine


def aggregateSeasonalAndAnnualValues(xCp, iSelect, nDim, nWindows, nStrata, nBoot):
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

    # Index using iSelect which could be a mask or a set of indices
    xCpSelect = index_or_mark_array_update(xCpSelect, iSelect, xCp)

    # Aggregate values based on dimensions
    if nDim == 2:
        xCpGF = xCpSelect  # naming convention
        # xCp shape = [nWindows, nBoot]
        CpA = np.nanmean(xCpGF, axis=0)
        nA = np.sum(~np.isnan(xCpSelect), axis=0)
    elif nDim == 3:
        xCpGF = xCpSelect  # Naming convention
        # xCp shape = [nWindows, nStrata, nBoot]
        # reshape => (nWindows*nStrata, nBoot) in column-major
        xCpGF_reshaped = np.reshape(xCpGF, (nWindows * nStrata, nBoot), order="F")
        CpA = np.nanmean(xCpGF_reshaped, axis=0)
        nA = np.sum(~np.isnan(xCpGF_reshaped), axis=0)
    else:
        raise ValueError("Invalid number of dimensions: Expected 2D or 3D Stats.")

    return CpA, nA, xCpSelect


def aggregateSeasonalMeans(
    mt: NDArray,
    Cp: NDArray,
    xmt: NDArray,
    iSelect: NDArray,
    nWindows: int,
    nStrata: int,
    nBoot: int,
) -> Tuple[NDArray, NDArray]:
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
            f"to shape ({nWindows}, {nStrata * nBoot})."
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


def updateSelectedIndices(
    iSelect: np.ndarray, iOut: np.ndarray, fSelect: np.ndarray, fOut: np.ndarray
) -> (np.ndarray, int, np.ndarray):
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


def updateModes(fModeX: np.ndarray, iOut: np.ndarray):
    """
    Recalculates mode indices after removing outliers, mirroring updateModes.m.

    Parameters
    ----------
    fModeX : np.ndarray
        Array indicating whether each point belongs to the mode (1) or not
        (0/NaN). This array is modified in-place, setting the elements
        at indices iOut to np.nan.
    iOut : np.ndarray
        Array of outlier indices to remove from the mode.

    Returns
    -------
    iModeX : np.ndarray
        Updated indices where fModeX is exactly 1.
    nModeX : int
        Number of such indices.
    """
    # Mark outlier positions in fModeX as NaN
    fModeX[iOut] = np.nan

    # Find indices where fModeX is exactly 1
    iModeX = np.where(fModeX == 1)[0]

    # Count them
    nModeX = len(iModeX)

    return iModeX, nModeX
