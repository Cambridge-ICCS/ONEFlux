from typing import Tuple, Sequence, Dict, Union, Any
from oneflux_steps.ustar_cp_python.fcBin import fcBin
from oneflux_steps.ustar_cp_python.utilities import prctile
from oneflux_steps.ustar_cp_python.fcDatenum import datenum
from oneflux_steps.ustar_cp_python.fcDatevec import fcDatevec
from oneflux_steps.ustar_cp_python.fcDoy import fcDoy
from oneflux_steps.ustar_cp_python.cpdFindChangePoint_functions import cpdFindChangePoint20100901
import numpy as np
from scipy.stats import pearsonr
import copy


def cpdEvaluateUStarTh4Season20100901(
    t: np.ndarray,
    NEE: np.ndarray,
    uStar: np.ndarray,
    T: np.ndarray,
    fNight: np.ndarray,
    fPlot: int,
    cSiteYr: str
) -> Tuple[np.ndarray, Any, np.ndarray, Any]:
    """
    cpdEvaluateUStarTh4Season20100901

    Estimates uStarTh for one year of data using change-point detection (cpd) methods
    within the general framework of the Papale et al. (2006) uStarTh evaluation method.
        
    Originally written (in MATLAB) by Alan Barr 15 Jan 2010.
    
    Parameters
    ----------
    t : np.ndarray
        Time vector (length >= one year of data).
    NEE : np.ndarray
        Net ecosystem exchange data corresponding to times in t.
    uStar : np.ndarray
        Friction velocity data corresponding to times in t.
    T : np.ndarray
        Temperature data corresponding to times in t.
    fNight : np.ndarray
        Vector specifying daytime (0) or nighttime (1) for each time step.
    fPlot : int
        Scalar flag for plotting (0 or 1). In MATLAB, if set to 1, various plots are produced.
        Here it is retained but not used directly unless placeholders or external plot calls are defined.
    cSiteYr : str
        Text string used in plot titles (if fPlot is enabled).

    Returns
    -------
    Cp2 : np.ndarray
        2D array (nSeasons x nStrataX) containing change-point (uStarTh) estimates
        from the 2-parameter operational model. Set to NaN if not enough data or
        if the model fails.
    Stats2 : Any
        2D array or nested structure with cpd statistics for the 2-parameter model.
        Its dimensions match (nSeasons x nStrataX).
    Cp3 : np.ndarray
        2D array (nSeasons x nStrataX) containing change-point (uStarTh) estimates
        from the 3-parameter diagnostic model. Set to NaN if not enough data or
        if the model fails.
    Stats3 : Any
        2D array or nested structure with cpd statistics for the 3-parameter model.
        Its dimensions match (nSeasons x nStrataX).

    Notes
    -----
    - The year of data is stratified by time of year (seasons) and temperature bins.
    - For each stratum, two change-point models (2-parameter operational, 3-parameter diagnostic)
      estimate the friction velocity threshold, uStarTh.
    """

    # ---------------------------
    # 0) Shallow copies of inputs
    # ---------------------------
    tCopy = t.copy()
    NEECopy = NEE.copy()
    uStarCopy = uStar.copy()
    TCopy = T.copy()
    fNightCopy = fNight.copy()

    # -------------------------------------
    # 1) Define basic partitioning constants
    # -------------------------------------
    nSeasons = 4
    nStrataN = 4
    nStrataX = 8
    nBins = 50

    # -----------------------------------------
    # 2) Initialize parameters and filter data
    # -----------------------------------------
    nt, m, EndDOY, nPerBin, nN = initializeParameters(
        tCopy, nSeasons, nStrataN, nBins
    )

    # filterInvalidPoints => (uStar, itAnnual, ntAnnual)
    uStarFiltered, itAnnual, ntAnnual = filterInvalidPoints(
        uStarCopy, fNightCopy, NEECopy, TCopy
    )

    # ----------------------------------------
    # 3) Initialize Cp2, Cp3, Stats2, Stats3
    # ----------------------------------------
    Cp2 = np.full((nSeasons, nStrataX), np.nan)
    Cp3 = np.full((nSeasons, nStrataX), np.nan)
    Stats2, Stats3 = initializeStatistics(nSeasons, nStrataX)

    # If not enough data, return now
    if ntAnnual < nN:
        return Cp2, Stats2, Cp3, Stats3

    # -------------------------------------------------
    # 4) Reorder and preprocess data (like "move Dec...")
    # -------------------------------------------------
    (
        tReordered,
        TReordered,
        uStarReordered,
        NEEReordered,
        fNightReordered,
        itAnnualReordered,
        ntAnnualReordered
    ) = reorderAndPreprocessData(
        tCopy, TCopy, uStarFiltered, NEECopy, fNightCopy, EndDOY, m, nt
    )


    # -----------------------------------------------------
    # 5) Adjust number of seasons based on actual good data
    # -----------------------------------------------------
    nPerSeason = round(ntAnnualReordered / nSeasons)
    nSeasons = round(ntAnnualReordered / nPerSeason)
    nPerSeason = ntAnnualReordered / nSeasons
    nPerSeason = round(nPerSeason)


    
    # ------------------------------------------------------------
    # 6) Stratify data in time (by season) and by temperature bins
    # ------------------------------------------------------------
    iPlot = 0
    if fPlot == 1:
        # Placeholder for figure creation call
        iPlot = 0

    for iSeason in range(nSeasons):
        # get the season indices

        jtSeason = computeSeasonIndices(iSeason, nSeasons, nPerSeason, ntAnnualReordered)
        itSeason = itAnnualReordered[jtSeason]
        ntSeason = len(itSeason)

        # compute number of strata
        nStrata = computeStrataCount(ntSeason, nBins, nPerBin, nStrataN, nStrataX)

        # temperature thresholds
        TTh = computeTemperatureThresholds(TReordered, itSeason, nStrata)

        for iStrata in range(nStrata):
            cPlotLocal, iPlot = None, None

            itStrata = findStratumIndices(TReordered, itSeason, TTh, iStrata)

            # bin the data
            nTemp, muStar, mNEE = fcBin(
                uStarReordered[itStrata], NEEReordered[itStrata], np.asarray([]), nPerBin
            )

            # Perform the change-point detection
            xCp2, xs2, xCp3, xs3 = cpdFindChangePoint20100901(
                muStar, mNEE, fPlot, cPlotLocal
            )

            # Additional stats
            nTemp, muStarT, mT = fcBin(
                uStarReordered[itStrata], TReordered[itStrata], np.asarray([]), nPerBin
            )
            rCorr, pCorr = corrcoef_with_pvalues(muStarT, mT)
            to_save = [xs2, tReordered, rCorr, pCorr, TReordered, itStrata]

            xs2 = addStatisticsFields(xs2, tReordered, rCorr, pCorr, TReordered, itStrata)
            xs3 = addStatisticsFields(xs3, tReordered, rCorr, pCorr, TReordered, itStrata)

            # Store results in the arrays
            Cp2[iSeason, iStrata] = xCp2
            Stats2[iSeason][iStrata] = xs2

            Cp3[iSeason, iStrata] = xCp3
            Stats3[iSeason][iStrata] = xs3

    return Cp2, Stats2, Cp3, Stats3


def reorderAndPreprocessData(
    t: np.ndarray, 
    T: np.ndarray, 
    u_star: np.ndarray, 
    NEE: np.ndarray, 
    f_night: np.ndarray, 
    end_doy: int, 
    m: np.ndarray, 
    nt: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Reorder and preprocess data based on specified conditions.

    Parameters:
        t (np.ndarray): Time array.
        T (np.ndarray): Temperature array.
        u_star (np.ndarray): Friction velocity array.
        NEE (np.ndarray): Net ecosystem exchange array.
        f_night (np.ndarray): Flag for nighttime data.
        end_doy (int): Last day of the year in day-of-year format.
        m (np.ndarray): Array of month indices.
        nt (int): Total number of data points.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
            - Updated time array.
            - Updated temperature array.
            - Updated friction velocity array.
            - Updated net ecosystem exchange array.
            - Updated nighttime flag array.
            - Indices of annual data.
            - Number of annual data points.
    """
    # Make copies of input arrays
    t = t.copy()
    T = T.copy()
    u_star = u_star.copy()
    NEE = NEE.copy()
    f_night = f_night.copy()
    m = m.copy()

    # Find indices where month equals December (12)
    it_d = np.where(m == 12)[0]

    # Determine reordering indices
    min_it_d = np.min(it_d)
    it_reorder = np.concatenate((np.arange(min_it_d, nt), np.arange(0, min_it_d)))

    # Adjust time for December and reorder arrays
    t[it_d] -= end_doy
    t = t[it_reorder]
    T = T[it_reorder]
    u_star = u_star[it_reorder]
    NEE = NEE[it_reorder]
    f_night = f_night[it_reorder]

    # Find valid annual data indices and their count
    it_annual = np.nonzero((f_night == 1) & ~np.isnan(NEE + u_star + T))[0]
    nt_annual = len(it_annual)

    return t, T, u_star, NEE, f_night, it_annual, nt_annual


def filterInvalidPoints(
    u_star: np.ndarray,
    f_night: np.ndarray,
    nee: np.ndarray,
    t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Filter invalid points from u_star and determine annual indices based on conditions.

    Parameters:
    u_star (np.ndarray): Array of u_star values.
    f_night (np.ndarray): Binary array indicating nighttime periods (1 for night, 0 for day).
    nee (np.ndarray): Array of net ecosystem exchange (NEE) values.
    t (np.ndarray): Array of time values.

    Returns:
    Tuple[np.ndarray, np.ndarray, int]:
        - Filtered u_star array with invalid values replaced by NaN.
        - Indices of valid annual points satisfying the conditions.
        - Number of valid annual points.
    """
    # Identify and filter invalid points in u_star (values outside the range [0, 3])
    invalid_indices = np.where((u_star < 0) | (u_star > 3))
    u_star[invalid_indices] = np.nan

    # Find valid annual indices satisfying the given conditions
    valid_annual_indices = np.where((f_night == 1) & ~np.isnan(nee + u_star + t))[0]
    num_valid_annual = len(valid_annual_indices)

    return u_star, valid_annual_indices, num_valid_annual


def addStatisticsFields(
    stats: Dict[str, float],
    t: np.ndarray, 
    r: np.ndarray, 
    p: np.ndarray, 
    T: np.ndarray, 
    itStrata: np.ndarray
) -> Dict[str, float]:
    """
    Calculate various expected values based on input arrays and indices.

    Parameters
    ----------
    t : np.ndarray
        One-dimensional array of values,  representing time.
    T : np.ndarray
        One-dimensional array of values, representing temperature.
    r : np.ndarray
        Two-dimensional array from which values for `expected_ruStarVsT` are extracted.
    p : np.ndarray
        Two-dimensional array from which values for `expected_puStarVsT` are extracted.
    itStrata : Sequence[int]
        Sequence of indices used to subset arrays `t` and `T`.


    Returns
    -------
    Dict[str, float]
        A dictionary containing:
        - "expected_mt": Mean of `t[itStrata]`
        - "expected_ti": The first element of `t[itStrata]`
        - "expected_tf": The last element of `t[itStrata]`
        - "expected_ruStarVsT": Extracted scalar value from `r[1][0]`
        - "expected_puStarVsT": Extracted scalar value from `p[1][0]`
        - "expected_mT": Mean of `T[itStrata]`
        - "expected_ciT": Half the difference between the 2.5th and 97.5th percentiles of `T[itStrata]`
    
    """

    expected_mt = np.mean(t[itStrata])
    expected_ti = t[itStrata[0]]
    expected_tf = t[itStrata[-1]]
    expected_ruStarVsT = r[1][0]
    expected_puStarVsT = p[1][0]
    expected_mT = np.mean(T[itStrata])
    ciT_vals = prctile(T[itStrata], [2.5, 97.5])
    expected_ciT = 0.5 * (np.diff(ciT_vals)[0])

    stats.update({
        "mt": expected_mt,
        "ti": expected_ti,
        "tf": expected_tf,
        "ruStarVsT": expected_ruStarVsT,
        "puStarVsT": expected_puStarVsT,
        "mT": expected_mT,
        "ciT": expected_ciT
    })

    return stats


def findStratumIndices(
    T: np.ndarray, 
    itSeason: Union[np.ndarray, np.ndarray],
    TTh: np.ndarray,
    iStrata: int
) -> np.ndarray:
    """
    Determine the indices within a specified range of T, and then intersect them with
    a given seasonal index set.

    The function first creates a boolean mask selecting the elements of T that lie between
    TTh[iStrata] and TTh[iStrata + 1]. It then extracts the indices of these elements, and
    intersects them with `itSeason` to produce the final set of indices.

    Parameters
    ----------
    T : np.ndarray
        A one-dimensional array of values (e.g., times, temperatures).
    TTh : np.ndarray
        A one-dimensional array of threshold values, used to define intervals in T.
    iStrata : int
        Index specifying which interval in TTh to use. The interval is defined as 
        [TTh[iStrata], TTh[iStrata + 1]].
    itSeason : Sequence[int] or np.ndarray
        A set of indices representing a particular season or subset of T.

    Returns
    -------
    np.ndarray
        An array of indices within the specified interval and season.

    Notes
    -----
    - The interval is inclusive of the endpoints.
    - `itSeason` is expected to be a set or list of valid indices for T.
    """

    # Create a boolean mask to select elements of T in the given interval
    mask = (T >= TTh[iStrata]) & (T <= TTh[iStrata + 1])

    # Extract indices where the condition is true
    itStrata_indices = np.nonzero(mask)[0]

    # Intersect the selected indices with itSeason
    expected_itStrata = np.intersect1d(itStrata_indices, itSeason)

    return expected_itStrata


def computeStrataCount(nt_season: int, 
                         n_bins: int, 
                         n_per_bin: int, 
                         n_strata_n: int, 
                         n_strata_x: int) -> int:
    """
    Compute the number of strata for a given season length and bin configuration.

    The computation is based on dividing the total number of time steps (nt_season) 
    by the product of the number of bins (n_bins) and the number of time steps per bin (n_per_bin).
    The resulting number of strata is then constrained between minimum (n_strata_n) 
    and maximum (n_strata_x) values.

    Parameters
    ----------
    nt_season : int
        Total number of time steps in a season.
    n_bins : int
        Number of bins used for stratification.
    n_per_bin : int
        Number of time steps per bin.
    n_strata_n : int
        Minimum allowable number of strata.
    n_strata_x : int
        Maximum allowable number of strata.

    Returns
    -------
    int
        The computed number of strata, ensuring it falls within the specified bounds.
    """
    n_strata = nt_season // (n_bins * n_per_bin)
    n_strata = max(n_strata, n_strata_n)
    n_strata = min(n_strata, n_strata_x)
    return n_strata

    
def computeSeasonIndices(i_season: int, 
                           n_seasons: int, 
                           n_per_season: int, 
                           nt_annual: int) -> range:
    """
    Compute the time indices for a given season within an annual cycle.

    The year is divided into `n_seasons` seasons, each ideally consisting of 
    `n_per_season` time steps. The function returns the indices for the 
    `i_season`-th season. For the last season, if the total annual length (`nt_annual`)
    is not perfectly divisible by `n_seasons`, it extends to the end of the year.

    Parameters
    ----------
    i_season : int
        The season index for which to compute the indices (1-based).
    n_seasons : int
        The total number of seasons in a year.
    n_per_season : int
        The number of time steps per season, ideally. Note that if the total 
        (`nt_annual`) is not divisible by `n_seasons`, the last season will 
        include the remainder.
    nt_annual : int
        The total number of time steps in a year.

    Returns
    -------
    range
        A range object representing the indices for the specified season.
    """
    n_per_season = int(n_per_season)
    if i_season == 0:
        # First season
        return range(0, n_per_season)
    elif i_season +1 == n_seasons: # +1 as 3rd index is 4th season
        # Last season, possibly extended to cover the remainder
        start = (n_seasons - 1) * n_per_season
        return range(start, nt_annual)
    else:
        # Intermediate seasons
        start = (i_season) * n_per_season
        end = (i_season+1) * n_per_season
        return range(start, end)


def computeTemperatureThresholds(
    T: np.ndarray, 
    it_season: Sequence[int], 
    n_strata: int
) -> np.ndarray:
    """
    Compute temperature thresholds by dividing the specified seasonal subset of 
    temperatures into `n_strata` strata based on percentile values. The resulting 
    thresholds will include the 0th percentile (minimum), the 100th percentile (maximum),
    and evenly spaced percentile boundaries in between.

    Parameters
    ----------
    T : np.ndarray
        A 1D array of temperature values.
    it_season : Sequence[int]
        Indices specifying which elements of `T` belong 
        to the season of interest.
    n_strata : int
        The number of strata to divide the temperatures into.

    Returns
    -------
    np.ndarray
        A 1D array of temperature thresholds, representing the percentile boundaries 
        from 0% to 100% inclusive. The array will have a length of `n_strata + 1`.
    """
    
    percentiles = np.linspace(0, 100, n_strata + 1)
    # Extract the seasonal temperatures and compute their percentiles
    T_season = T[it_season]
    TTh = prctile(T_season, percentiles)

    return TTh


def initializeParameters(
    t: np.ndarray,
    nSeasons: int,
    nStrataN: int,
    nBins: int
) -> Tuple[int, np.ndarray, float, int, int]:
    """
    initializeParameters

    Derives basic time information from an input time array t (e.g., serial
    datenum or timestamp) and initializes binning parameters for subsequent
    computations.

    Parameters
    ----------
    t : np.ndarray
        Time array (1D). Must be numeric. 
    nSeasons : int
        Number of seasons to consider.
    nStrataN : int
        Number of strata for each season.
    nBins : int
        Number of bins used in the computations.

    Returns
    -------
    nt : int
        Length of t.
    m : np.ndarray
        Month array derived from fcDatevec(t).
    EndDOY : float
        Day of year (DOY) for the last day of the median year in the data.
    nPerBin : int
        Number of time steps per bin, adjusted based on sampling frequency.
    nN : int
        Total number of points across seasons, strata, and bins: nSeasons * (nStrataN * nBins * nPerBin).

    Example
    -------
    >>> t = np.linspace(737060.0, 737389.0, 24*365/2)  # Example time array
    >>> nt, m, EndDOY, nPerBin, nN = initializeParameters(t, 4, 2, 3)
    """

    # Derive basic time information
    nt = len(t)
    y, m, _, _, _, _ = fcDatevec(t)  
    iYr = int(np.median(y))          # median year
    EndDOY = fcDoy(datenum(iYr, 12, 31.5))  

    # Estimate sampling frequency
    # e.g., daily data => nPerDay=1, or 30-min data => nPerDay=48, etc.
    nPerDay = int(round(1 / np.nanmedian(np.diff(t))))

    # Define default
    nPerBin = 5

    # Adjust binning based on sampling frequency
    if nPerDay == 24:
        nPerBin = 3
    elif nPerDay == 48:
        nPerBin = 5

    # Number of points per season for 'n' dimension
    nPerSeasonN = nStrataN * nBins * nPerBin
    nN = nSeasons * nPerSeasonN

    return nt, m, EndDOY, nPerBin, nN


def initializeStatistics(nSeasons: int, nStrataX: int) -> Tuple[Any, Any]:
    """
    initializeStatistics

    Creates two 2D arrays (Stats2, Stats3) of the same shape [nSeasons x nStrataX].
    Each entry in these arrays is a shallow copy of a base stats object (StatsMT)
    generated by generate_statsMT(). This replicates the MATLAB code logic, but uses
    Python's 0-based indexing and shallow copies.

    Parameters
    ----------
    nSeasons : int
        Number of seasons.
    nStrataX : int
        Number of strata for the x dimension.

    Returns
    -------
    Stats2 : 2D list (or array) of shape [nSeasons x nStrataX]
        Each element is a shallow copy of StatsMT.
    Stats3 : 2D list (or array) of shape [nSeasons x nStrataX]
        Each element is a shallow copy of StatsMT.

    Notes
    -----
    -
    """

    # Retrieve the base stats object (placeholder function).
    from oneflux_steps.ustar_cp_python.cpdBootstrap import generate_statsMT
    StatsMT = generate_statsMT()

    # Construct Stats2, Stats3 as 2D arrays (lists of lists here)
    # Each element is a shallow copy of StatsMT
    Stats2 = [[copy.copy(StatsMT) for _ in range(nStrataX)] for _ in range(nSeasons)]
    Stats3 = [[copy.copy(StatsMT) for _ in range(nStrataX)] for _ in range(nSeasons)]

    return Stats2, Stats3


def corrcoef_with_pvalues(muStar: np.ndarray, mT: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the correlation coefficient matrix (R) and p-values (P) 
    for two 1D NumPy arrays muStar and mT.

    Parameters:
    muStar (np.ndarray): First 1D array of data.
    mT (np.ndarray): Second 1D array of data.

    Returns:
    Tuple[np.ndarray, np.ndarray]:
        - R: Correlation coefficient matrix.
        - P: P-values matrix corresponding to significance levels of correlations.
    """
    # Stack the arrays to create a 2D matrix similar to MATLAB's corrcoef input
    A = np.column_stack((muStar, mT))
    
    n = A.shape[1]
    R = np.corrcoef(A, rowvar=False)  # Compute correlation matrix
    P = np.ones((n, n))  # Initialize P matrix with ones

    # Compute p-values for each pair
    for i in range(n):
        for j in range(i + 1, n):  # Only compute upper triangle
            r, p = pearsonr(A[:, i], A[:, j])
            R[i, j] = R[j, i] = r  # Fill both symmetric parts
            P[i, j] = P[j, i] = p  # Fill both symmetric parts

    return R, P