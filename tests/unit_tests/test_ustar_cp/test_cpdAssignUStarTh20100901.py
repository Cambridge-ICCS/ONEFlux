import pytest
import matlab.engine
import numpy as np


@pytest.fixture(scope="module")
def mock_data(test_engine, nt=300, tspan=(0, 1), uStar_pars=(0.1, 3.5), T_pars=(-10, 30), fNight=None):
    """
    Fixture to generate mock time series data for testing purposes. This fixture
    creates a set of synthetic data typically used in environmental studies,
    such as Net Ecosystem Exchange (NEE), uStar values, temperature, and day/night flags.

    Args:
        nt (int, optional): Number of time points to generate in the series.
                            Defaults to 300.
        tspan (tuple, optional): Start and end time for the time vector.
                                 Defaults to (0, 1).
        uStar_pars (tuple, optional): Minimum and maximum values for the random
                                      generation of u* values. Defaults to (0.1, 3.5).
        T_pars (tuple, optional): Minimum and maximum values for the random
                                  generation of temperature values. Defaults to (-10, 30).
        fNight (array-like or None, optional): Binary values indicating day (0) or
                                               night (1). If None, a random assignment
                                               is made. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - t (np.ndarray): Time vector of length `nt`.
            - NEE (np.ndarray): Randomly generated Net Ecosystem Exchange values.
            - uStar (np.ndarray): Randomly generated u* values.
            - T (np.ndarray): Randomly generated temperature values.
            - fNight (np.ndarray): Array indicating day (0) or night (1) conditions.
    """
    fPlot = 0
    nBoot = 10
    cSiteYr = "Site_2024"
    t = np.linspace(*tspan, nt)  # Generate time vector
    uStar = np.random.uniform(*uStar_pars, size=nt)  # u* values between typical ranges
    NEE = np.random.uniform(-10, 10, size=nt)  # Random NEE values
    T = np.random.uniform(*T_pars, size=nt)  # Temperature values
    if fNight is None:
        fNight = np.random.choice([0, 1], size=nt)  # Randomly assign day/night
    else:
        fNight = np.resize(fNight, nt)

    Cp2, Stats2, Cp3, Stats3 = test_engine.cpdBootstrapUStarTh4Season20100901(
        t,NEE,uStar,T,fNight,fPlot,cSiteYr,nBoot,jsonencode=[1,3],nargout=4
    )
    return Stats2, fPlot, cSiteYr


# def test_cpdAssignUStarTh20100901_basic(test_engine, mock_data):
#     stats, fPlot, cSiteYr = mock_data

#     # Call MATLAB function
#     CpA, nA, tW, CpW, cMode, cFailure, fSelect, sSine, FracSig, FracModeD, FracSelect = test_engine.cpdAssignUStarTh20100901(stats, fPlot, cSiteYr, jsondecode=[0], nargout=11)

#     # Assertions
#     assert isinstance(CpA, matlab.double), "CpA should be a MATLAB double array"
#     assert isinstance(nA, matlab.double), "nA should be a MATLAB double array"
#     assert isinstance(tW, matlab.double), "tW should be a MATLAB double array"
#     assert isinstance(CpW, matlab.double), "CpW should be a MATLAB double array"
#     assert isinstance(cMode, str), "cMode should be a string"
#     assert isinstance(cFailure, str), "cFailure should be a string"
#     assert isinstance(fSelect, matlab.logical), "fSelect should be a MATLAB logical array"
#     assert isinstance(sSine, matlab.double), "sSine should be a MATLAB double array"
#     assert isinstance(FracSig, float), "FracSig should be a float"
#     assert isinstance(FracModeD, float), "FracModeD should be a float"
#     assert isinstance(FracSelect, float), "FracSelect should be a float"


def test_cpdAssignUStarTh20100901_edge_cases(test_engine, mock_data):
    def set_attr(struct_array, key, val):
        np.vectorize(lambda x: x.update({key: val}))(struct_array)

    mock_stats, fPlot, cSiteYr = mock_data
    edge_stats = mock_stats.copy()

    # Case 1: All significant change points
    set_attr(edge_stats, "p", 0)
    assert edge_stats[0][0][0]['p'] == 0

    results_all_sig = test_engine.cpdAssignUStarTh20100901(edge_stats, 0, "AllSig_2024", jsondecode=[0], nargout=11)

    # Case 2: No significant change points
    set_attr(edge_stats, "p", 1)
    assert edge_stats[0][0][0]['p'] == 1

    results_no_sig = test_engine.cpdAssignUStarTh20100901(edge_stats, 0, "NoSig_2024", jsondecode=[0], nargout=11)

    # Assertions for edge cases
    assert len(results_all_sig[0]) > 0, "Should produce results for all significant change points"
    assert len(results_no_sig[5]) > 0, "Should produce a failure message for no significant change points"

@pytest.mark.parametrize(
    "x_norm_x, threshold, expected_f_out, expected_i_out",
    [
        ([0.5, 1.2, 3.5, 0.1, 2.8], 2.0, [[False, False, True, False, True]], [[3.0, 5.0]]),  # Basic test
        ([0.1, 0.2, 0.3], 1.0, [[False, False, False]], [[]]),  # No outliers
        ([3.1, 2.9, 3.5], 2.0, [[True, True, True]], [[1, 2, 3]]),  # All outliers
        ([], 2.0, [], []),  # Empty input case
        ([-3, -2, -1, 0, 1, 2, 3], -1.0, [[False, False, False, True, True, True, True]], [[4.0, 5.0, 6.0, 7.0]]),  # Negative threshold
    ]
)
def test_identify_outliers(test_engine, x_norm_x, threshold, expected_f_out, expected_i_out):
    """Test MATLAB's identifyOutliers function from Python using MATLAB Engine."""


    f_out, i_out = test_engine.identifyOutliers(test_engine.convert(x_norm_x), test_engine.convert(threshold), nargout=2)

    assert test_engine.equal(f_out, test_engine.convert(expected_f_out)), "Boolean outlier array does not match expected"
    assert test_engine.equal(i_out, test_engine.convert(expected_i_out, index="to_python")), "Index output does not match expected"

@pytest.mark.parametrize(
    "xCp, iSelect, nDim, nWindows, nStrata, nBoot, expected_CpA, expected_nA, expected_xCpSelect",
    [
        ([1.0, 2.0, np.nan, 3.0, np.nan, 4.0]
          , [True, True, False, True, False, True]
          , 2, 0, 0, 0
           # expected
          , 2.5, 4.0, [1.0,2.0,np.nan,3.0,np.nan,4.0]
         )
        , ([[1.0, np.nan, 3.0],[4.0, 5.0, np.nan]]
          , ([[True, False, True],[False, True, False]])
          , 3, 2, 1, 3
          # expected
          , [1.0, 5.0, 3.0], [1.0, 1.0, 1.0], [[1, np.nan, 3.0], [np.nan, 5.0, np.nan]])
        ]
)

def test_aggregate_3d_case(test_engine, xCp, iSelect, nDim, nWindows, nStrata, nBoot, expected_CpA, expected_nA, expected_xCpSelect):
    """Test function for aggregateSeasonalAndAnnualValues 2D and 3D cases."""


    # Run MATLAB function
    CpA, nA, xCpSelect = test_engine.aggregateSeasonalAndAnnualValues(
        test_engine.convert(xCp), test_engine.convert(iSelect), nDim, nWindows, test_engine.convert(nStrata), test_engine.convert(nBoot), nargout=3
    )

    assert test_engine.equal(CpA, test_engine.convert(expected_CpA))
    assert test_engine.equal(nA, test_engine.convert(expected_nA))
    assert test_engine.equal(xCpSelect, test_engine.convert(expected_xCpSelect))