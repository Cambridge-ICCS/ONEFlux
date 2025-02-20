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
        In MATLAB, can be 2D [nWindows, nBoot] or 3D [nWindows, nStrata, nBoot].
        In Python, pass a nested list or NumPy array with the same shape.
    iSelect : array-like of int
        1-based linear indices in MATLAB's column-major ordering that should be selected.
        We subtract 1 and interpret them in the same column-major order in Python.
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
    # # Convert inputs to float arrays
    # xCp = np.asarray(xCp, dtype=float)

    # Prepare an output array (same shape) filled with NaNs
    xCpSelect = np.full_like(xCp, np.nan, dtype=float)

    # Flatten both arrays in column-major (Fortran) order to replicate MATLAB indexing
#    xCp_flat_F = xCp.flatten(order='F')
#    xCpSelect_flat_F = xCpSelect.flatten(order='F')

    iSelect_array = np.asarray(iSelect, dtype=int)

    # Assign the selected change points in the flattened array
    # mask xCp using iSelect_array
    xCpSelect = xCp.copy()
    for i in range(len(xCp)):
        if iSelect_array[i] == 0:
            xCpSelect[i] = np.nan
    
    # Reshape back to original shape (column-major)
    #xCpSelect = xCpSelect_flat_F.reshape(xCp.shape, order='F')
    xCpGF = xCpSelect  # Just like the MATLAB code

    # Aggregate values based on dimensions
    if nDim == 2:
        # xCp shape = [nWindows, nBoot]
        # MATLAB default nanmean(xCpGF) => mean across axis=0 in Python
        CpA = np.nanmean(xCpGF, axis=0)
        nA  = np.sum(~np.isnan(xCpSelect), axis=0)
    elif nDim == 3:
        # xCp shape = [nWindows, nStrata, nBoot]
        # reshape => (nWindows*nStrata, nBoot) in column-major
        xCpGF_reshaped = xCpGF.reshape(nWindows * nStrata, nBoot, order='F')
        CpA = np.nanmean(xCpGF_reshaped, axis=0)
        nA  = np.sum(~np.isnan(xCpGF_reshaped), axis=0)
    else:
        raise ValueError("Invalid number of dimensions: Expected 2D or 3D Stats.")

    return CpA, nA, xCpSelect
