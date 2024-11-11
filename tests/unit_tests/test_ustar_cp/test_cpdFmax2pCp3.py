import pytest
import numpy as np
import matlab.engine

@pytest.mark.parametrize(
    "Fmax, n, expected",
    [
        (np.nan, 10, np.nan),     # Test case with Fmax = NaN
        (10, np.nan, np.nan),     # Test case with n = NaN
        (10, 5, np.nan),          # Test case with n < 10
        ],
)
def test_cpdFmax2pCp3_return_nan(matlab_engine, Fmax, n, expected):
    """
    Test the cpdFmax2pCp3 MATLAB function for cases where `nan` is returned.
    """
    # Handle NaN inputs for MATLAB
    Fmax_matlab = matlab.double([Fmax]) if not np.isnan(Fmax) else matlab.double([float('nan')])
    n_matlab = matlab.double([n]) if not np.isnan(n) else matlab.double([float('nan')])

    # Call the MATLAB function
    result = matlab_engine.cpdFmax2pCp3(Fmax_matlab, n_matlab)

    assert type(result) == type(expected), f"Result should be a {type(expected)}, got {type(result)}"


@pytest.mark.parametrize(
    "Fmax, n, expected",
    [
        (5, 20, 0.4393355161666135),       # Fmax below critical value
        (30, 1000, 2.7278340575698223e-05),# Fmax above critical value
        (11.5, 100, 0.0432111321429921),   # Fmax within critical range
    ],
)

def test_cpdFmax2pCp3_return_numeric(matlab_engine, Fmax, n, expected):
    """
    Test the cpdFmax2pCp3 MATLAB function for cases where numeric values are returned.
    """
    # Handle NaN inputs for MATLAB
    Fmax_matlab = matlab.double([Fmax]) if not np.isnan(Fmax) else matlab.double([float('nan')])
    n_matlab = matlab.double([n]) if not np.isnan(n) else matlab.double([float('nan')])

    # Call the MATLAB function
    result = matlab_engine.cpdFmax2pCp3(Fmax_matlab, n_matlab)

    assert type(result) == type(expected), f"Result should be a {type(expected)}, got {type(result)}"
    assert np.allclose(np.array(result), np.array(expected))