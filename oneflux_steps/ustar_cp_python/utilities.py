# A collection of utility functions that are used in the ustar_cp_python module.

import numpy as np
import json
from decimal import Decimal, ROUND_HALF_UP


def index_or_mark_array_update(
    A: np.ndarray, idx: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """
    index_or_mark_array_update(A, idx, B) performs something akin to `A[idx] = B`
    where we update the elements of `A` at indices `idx` according to `B`
    but where A and B are n-dimensional but idx is either an n-dimensional array
    of masking booleans, or a 1-dimensional array of indices into the flattened
    array A.

    Parameters:
    A : np.ndarray
        The input array.
    idx : np.ndarray
        The indices to use.

    Returns:
    np.ndarray
        The elements of the array at the specified indices.
    """

    if all(isinstance(i, np.bool_) for i in np.array(idx).flat):
        # idx is a boolean mask
        for i in range(len(A.flat)):
            if idx.flat[i]:
                A.flat[i] = B.flat[i]
    else:
        # Do column major flattening of A
        Acopy = A.copy().flatten(order="F")
        B = B.flatten(order="F")
        idx = idx.flatten()
        # Copy across the elements at the right indices
        for i in idx:
            Acopy[i] = B[i]
        # Reshape the data
        A = Acopy.reshape(A.shape, order="F")
    return A


def prctile(A: np.ndarray, p: float) -> float | np.ndarray:
    """
    Compute the p-th percentile of array A in a way that has
    (as far as we can tell) the same semantics MATLAB's percentile algorithm.

    Args:
        A (np.ndarray): Input 1D array.
        p (float): Desired percentile (0 to 100).

    Returns:
        float|np.ndarray: The computed percentile value.
    """
    A = A[~np.isnan(A)]  # Remove NaNs
    n = len(A)
    if n == 0:
        return np.nan

    A_sorted = np.sort(A)

    # Define percentiles at sorted element positions
    percentiles = 100 * (np.arange(0.5, n) / n)

    # Handle bounds explicitly
    # if p <= percentiles[0]:
    # return A_sorted[0]
    # elif p >= percentiles[-1]:
    # return A_sorted[-1]

    # Linear interpolation
    return np.interp(p, percentiles, A_sorted)


def prctile_hazen(a, q):
    """
    Compute the q-th percentile of the data along the specified axis.
    """
    q = np.asarray(q)
    if np.size(a) == 0:
        return np.full_like(q, np.nan)
    a = np.percentile(np.asarray(a), q, method="hazen")
    return a


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the dot product of two arrays.
    """
    try:
        return np.dot(a, b)
    except ValueError:
        return sum([x * y for x, y in zip(a, b)])


def floor(a: np.ndarray) -> np.ndarray:
    """
    Return the floor of the input, element-wise.
    """
    return np.asanyarray(a // 1).astype(int)


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the intersection of two arrays.

    Parameters:
    a : np.ndarray
        The first array to intersect

    b : np.ndarray
        The second array to intersect

    nargout : int
        The number of output arguments to return

    Returns:
    np.ndarray
        The intersection of the two arrays
    """
    from builtins import set

    c = sorted(set(a.flat) & set(b.flat))
    if isinstance(a, str):
        return "".join(c)
    elif isinstance(a, list):
        return c
    else:
        if (len(a.shape) > 1) and (a.shape[1] > 1):
            return transpose(np.array(c))
        else:
            if len(c) == 1:
                # If the result is a singleton, return it as a scalar
                return c[0]
            else:
                return np.array(c)
        # FIXME: the result is a column vector if
        # both args are column vectors; otherwise row vector
        # return np.array(c).reshape((1, -1) if a.shape[1] > 1 else (-1, 1))


def jsonencode(a):
    return a if isinstance(a, cellarray) else json.dumps(a)


def jsondecode(a):
    return a if isinstance(a, cellarray) else json.loads(a)


def ndims(a: int | float | np.ndarray) -> int:
    """
    Compute the number of dimensions on a piece of data

    Parameters:
    a : np.ndarray | int | float
        The data to compute the number of dimensions for

    Returns:
    int: The number of dimensions for the data

    Examples:
    >>> ndims(1)
    2
    >>> ndims(np.array([1,2,3]))
    2
    >>> ndims(np.array([[1,2,3]]))
    2
    >>> ndims(np.array([[[1,2,3]]]))
    3
    """
    # Scalars are treated as 2D (singleton) matrics by ndim
    if isinstance(a, int) or isinstance(a, float):
        return 2
    else:
        ndim = np.asarray(a).ndim
        if ndim < 2:
            return 2
        else:
            return ndim


def arange(start, stop, step=1, **kwargs):
    """
    >>> a=arange(1,10) # 1:10
    >>> size(a)
    matlabarray([[ 1, 10]])
    """
    expand_value = 1 if step > 0 else -1
    return np.arange(start, stop + expand_value, step, **kwargs)


def round_up(value: float) -> int:
    """
    Round a number to the nearest integer, with ties rounding away from zero.

    Parameters:
    value (float): The number to round.

    Returns:
    int: The rounded integer.
    """
    return int(Decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def size(a: np.ndarray, b=0, nargout=1) -> np.ndarray:
    """
    Return the size of an array.
    >>> size(np.array([[1,2],[3,4]]))
    np.ndarray([[2, 2]])

    >>> size(zeros(3,3)) + 1
    np.ndarray([[4, 4]])
    """
    s = np.asarray(a).shape
    if s == ():
        return 1 if b else (1,) * nargout
    # a is not a scalar
    try:
        if b:
            return s[b - 1]
        else:
            return np.squeeze(s)
    except IndexError:
        return 1


def squeeze(a: np.ndarray, axis=None) -> np.ndarray:
    """
    Remove single-dimensional entries from the shape of an array.
    """
    if axis is not None and a.shape[axis] != 1:
        return a
    if axis is None and a.ndim == 2 and a.shape[0] == 1:
        return a.reshape(-1)
    return np.squeeze(a, axis=axis)


def transpose(a: np.ndarray | list | float | int) -> np.ndarray:
    """
    A multi-purpose transpose function that
    - on 2-dimensions, does the usual matrix transpotision
    - on 1-dimensional data, converts a row vector to a column vector

    Note that a column vector already looks 2-dimensional, and so its
     transpose gives us back a row-vector (but as a matrix), thus
     this operation is not an involution.

    >>> transpose(np.array([[1,2],[3,4]]))
    array([[1, 3],
          [2, 4]])
    >>> transpose(np.array([1,2,3]))
    array([[1],
          [2],
          [3]])
    >>> transpose(transpose(np.array([1,2,3])))
    array([[1, 2, 3]])

    """
    if np.ndim(a) == 2:
        # if we have a tuple then
        # we have to convert it to a list
        # because tuples are immutable
        if isinstance(a, tuple):
            a = np.asarray(list(a))

        return a.transpose()
    elif np.ndim(a) == 1:
        return a.reshape(-1, 1)
    elif np.ndim(a) == 0:
        return a
    else:
        raise (
            "Transpose is not defined for arrays of dimension greater than 2, but given data of dimension "
            + str(np.ndim(a))
        )


def unique(a):
    """
    Return the unique elements of an array.
    """
    return np.unique(np.asarray(a))


# Plotting relates


def xlim(left=None, right=None):
    """
    Set the x limits of the current axes.
    """
    import matplotlib.pyplot as plt

    plt.xlim(left, right)


# Used by JSON encoding/decoding


class cellarray(np.ndarray):
    """
    Cell array corresponds to matlab ``{}``
    """

    def __new__(cls, a=[]):
        """
        Create a cell array and initialize it with a.
        Without arguments, create an empty cell array.
        Parameters:
        a : list, ndarray, etc.
        >>> a=cellarray([123,"hello"])
        >>> print(a.shape)
        (1, 2)
        >>> print(a[1])
        123
        >>> print(a[2])
        hello
        """
        return super().__new__(cls, a, dtype=object)


def nlinfit(xdata, ydata, f, initial_guess=None):
    """
    Perform a non-linear fit to the data using the provided initial guess.

    Parameters:
    xdata (np.ndarray): The independent variable data.
    ydata (np.ndarray): The dependent variable data.
    initial_guess (list, optional): Initial guess for the parameters. Defaults to None.
    Returns:
    np.ndarray: The fitted parameters.
    """
    from scipy.optimize import curve_fit

    # Define a local function for curve_fit: curve_fit expects
    # a callable f(t, b0, b1, b2) with first arg = x, subsequent = params
    def _fun_to_fit(xdataUpd, b0, b1, b2):
        return f(np.asarray([b0, b1, b2]), xdataUpd)

    if initial_guess is None:
        initial_guess = [1.0, 1.0, 1.0]

    # Perform the fit via non-linear regression
    popt, _ = curve_fit(
        _fun_to_fit,
        xdata,
        ydata,
        p0=initial_guess,
        nan_policy="omit",
        method="trf",
    )

    return popt
