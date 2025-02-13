# test_aggregateSeasonalMeans.py
import pytest
import numpy as np

@pytest.mark.parametrize(
    "mt, Cp, xmt, iSelect, nWindows, nStrata, nBoot, expected_tW, expected_CpW",
    [
        # -- Test Case 1: Simple data, single "window" scenario --
        (
            [10, 20, 30],          # mt
            [1.0, 2.0, 3.0],       # Cp
            [20],          # xmt
            [True, True, True],    # iSelect (bool mask)
            1,                     # nWindows
            1,                     # nStrata
            1,                     # nBoot
            [10.0],                # expected_tW (mock result; you must refine)
            [1.0],                 # expected_CpW (mock result)
        ),

        # -- Test Case 2: Multiple windows, partial selection --
        (
    [10, 20, 30, 40, 50],               # mt
    [1.0, 2.0, 4.0, 8.0, 16.0],         # Cp
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 
     100,110,120,130,140,150,160,170,180,190,
     200,210,220,230,240],  # <-- 24 elements for xmt
    [False, True, True, True, False],  # iSelect
    4,                                  # nWindows
    3,                                  # nStrata
    2,                                  # nBoot
    [25.0, 40.0],                       # expected_tW
    [3.0, 8.0],                         # expected_CpW
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
    """

    # Call the MATLAB function. Note nargout=2 to receive two outputs (tW, CpW)
    tW_mat, CpW_mat = test_engine.aggregateSeasonalMeans(
        test_engine.convert(mt),   # mt
        test_engine.convert(Cp),   # Cp
        test_engine.convert(xmt),  # xmt
        test_engine.convert(iSelect),
        test_engine.convert(nWindows),
        test_engine.convert(nStrata),
        test_engine.convert(nBoot),
        nargout=2
    )

    # Convert outputs from MATLAB arrays back to Python floats/arrays
    # MATLAB returns "double" arrays. We turn them into NumPy arrays.
    tW = np.array(tW_mat).flatten()
    CpW = np.array(CpW_mat).flatten()

    # Compare the results with the expected values (within a tolerance).
    # Adjust `rtol`, `atol` as needed, or do exact match if you expect integer results.
    assert np.allclose(tW, expected_tW, rtol=1e-5, atol=1e-8), \
           f"tW mismatch: got {tW}, expected {expected_tW}"
    assert np.allclose(CpW, expected_CpW, rtol=1e-5, atol=1e-8), \
           f"CpW mismatch: got {CpW}, expected {expected_CpW}"