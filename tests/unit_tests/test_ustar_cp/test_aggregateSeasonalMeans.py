import pytest
import numpy as np

@pytest.mark.parametrize(
    "mt, Cp, xmt, iSelect, nWindows, nStrata, nBoot, expected_tW, expected_CpW",
    [
        #
        # Test Case 1: Single window, select all points with numeric indices
        #
        (
            [10, 20, 30],          # mt
            [1.0, 2.0, 3.0],       # Cp
            [20],                  # xmt => length=1 => reshape(1,1*1)
            [1, 2, 3],            # iSelect as numeric indices (MATLAB is 1-based)
            1,                     # nWindows
            1,                     # nStrata
            1,                     # nBoot
            [20.0],                # expected_tW
            [2.0],                 # expected_CpW
        ),

        #
        # Test Case 2: Multiple windows, partial selection
        # Here iSelect = [2,3,4] means we are selecting the 2nd, 3rd, and 4th elements
        # in the arrays mt/Cp. xmt must have 4*(3*2) = 24 elements for reshape().
        #
        (
            [10, 20, 30, 40, 50],               # mt
            [1.0, 2.0, 4.0, 8.0, 16.0],         # Cp
            [
                10, 20, 30, 40, 50, 60, 70, 80, 90,
                100,110,120,130,140,150,160,170,180,190,
                200,210,220,230,240
            ],                                  # xmt => 24 elements
            [2, 3, 4],                          # iSelect => partial selection
            4,                                  # nWindows
            3,                                  # nStrata
            2,                                  # nBoot
            [20.0, 30.0, 30.0, 40.0],                       # expected_tW
            [2.0, 4.0, 4.0, 8.0],                         # expected_CpW (mock result)
        ),
    ]
)
def test_aggregateSeasonalMeans(test_engine, mt, Cp, xmt, iSelect,
                                nWindows, nStrata, nBoot,
                                expected_tW, expected_CpW):
    """
    Test the MATLAB function aggregateSeasonalMeans by calling it via
    the MATLAB Engine API for Python. We pass arrays from Python
    to MATLAB, execute the function, and verify the outputs.

    Key point:
    ----------
    - In the original cpdAssignUStarTh20100901.m code, `iSelect` is a
      numeric array of indices (found via `find(...)`) rather than a
      logical mask. This avoids the mixing of sort(...) and logical
      indexing that caused errors.
    - We ensure length(xmt) == nWindows * nStrata * nBoot so reshape(xmt, ...)
      is valid in MATLAB.
    """

    mt_matlab = test_engine.convert(mt)
    Cp_matlab = test_engine.convert(Cp)
    xmt_matlab = test_engine.convert(xmt)

    # Important: iSelect is now numeric. For MATLAB, it's 1-based indexing.
    # This means if iSelect=[2,3], we are selecting the 2nd, 3rd elements of mt/Cp.
    iSelect_test = test_engine.convert(iSelect, index = "to_python")

    nWindows_test = test_engine.convert(nWindows)
    nStrata_test = test_engine.convert(nStrata)
    nBoot_test = test_engine.convert(nBoot)

    # Call the function
    tW_mat, CpW_mat = test_engine.aggregateSeasonalMeans(
        mt_matlab,
        Cp_matlab,
        xmt_matlab,
        iSelect_matlab,
        nWindows_matlab,
        nStrata_matlab,
        nBoot_matlab,
        nargout=2
    )

    # Convert outputs to NumPy arrays
    tW = np.array(tW_mat).flatten()
    CpW = np.array(CpW_mat).flatten()

    # Compare the results with the expected values
    assert np.allclose(tW, expected_tW, rtol=1e-5, atol=1e-8), \
           f"tW mismatch: got {tW}, expected {expected_tW}"
    assert np.allclose(CpW, expected_CpW, rtol=1e-5, atol=1e-8), \
           f"CpW mismatch: got {CpW}, expected {expected_CpW}"
