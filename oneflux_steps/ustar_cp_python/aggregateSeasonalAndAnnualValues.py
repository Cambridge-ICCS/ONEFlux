import numpy as np

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
