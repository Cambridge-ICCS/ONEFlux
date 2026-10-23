import numpy as np


def fcx2rowvec(x: np.ndarray) -> np.ndarray:
    """
    Converts an input array `x` to a 1 x n row vector.

    Design note (row-major vs column-major)
    ---------------------------------------
    The project-wide design rules state that Python code should work in
    row-major order. However, this function intentionally flattens in
    **column-major (Fortran) order** to replicate MATLAB's linear-indexing
    convention. The function is used to mirror MATLAB's ``x(:)'`` idiom in
    code paths that still reason about the data as a MATLAB-style linear
    stream. See the longer note in ``fcx2colvec`` for why these two must
    be kept consistent and when they could be migrated to row-major.

    Args:
        x (np.ndarray): Input array to be reshaped.

    Returns:
        np.ndarray: Reshaped array as a row vector.

    Example:
        >>> import numpy as np
        >>> x = np.array([[1, 2], [3, 4]])
        >>> fcx2rowvec(x)
        array([[1, 3, 2, 4]])
    """
    return np.reshape(x, (1, -1), order="F")
