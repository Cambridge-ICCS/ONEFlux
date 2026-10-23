import numpy as np


def fcx2colvec(x: np.ndarray) -> np.ndarray:
    """
    Converts an input array `x` to an n x 1 column vector.

    Design note (row-major vs column-major)
    ---------------------------------------
    The project-wide design rules state that Python code should work in
    row-major order. However, this function intentionally flattens in
    **column-major (Fortran) order** to replicate MATLAB's linear-indexing
    convention. MATLAB's linear indices enumerate elements down columns
    first; several call sites in ``cpdAssignUStarTh`` produce index arrays
    (``selectedIndices``, ``modeDIndices``, ...) drawn from these flattened
    vectors and then reuse the same indices to address the original 3-D
    ``Stats``/``changePoint`` arrays via ``index_or_mark_array_update``
    (which also flattens with ``order='F'``). The pair of F-order
    flatten/unflatten operations must be kept in lock-step; changing one
    without the other will silently permute which cells of ``Stats`` the
    computed indices refer to.

    Therefore: keep ``order='F'`` here UNLESS the downstream indexing
    scheme in ``cpdAssignUStarTh`` is refactored to row-major at the same
    time. Same applies to ``fcx2rowvec``.

    Args:
        x (np.ndarray): Input array to be reshaped.

    Returns:
        np.ndarray: Reshaped array as a column vector.

    Example:
        >>> import numpy as np
        >>> x = np.array([[1, 2], [3, 4]])
        >>> fcx2colvec(x)
        array([[1],
               [3],
               [2],
               [4]])
    """
    return np.reshape(x, (-1, 1), order="F")
