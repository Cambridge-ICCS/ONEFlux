"""
Differential tests for MATLAB and Python of fcDatevec function
"""

from hypothesis import given, settings
from hypothesis.strategies import floats, lists
from oneflux_steps.ustar_cp_python.fcDatevec import fcDatevec
import pandas as pd


# Property-based tests for fcDatevec
# The size of the input `n` determines the size of the output as `n x 6`
@given(
    data=lists(
        floats(allow_infinity=False, min_value=-10000, max_value=10000),
        min_size=1,
        max_size=100,
    )
)
@settings(deadline=1000)
def test_differential(test_engine, data):
    # Call the function
    result = test_engine.fcDatevec(test_engine.convert(data), nargout=6)

    # Compare matlab and python
    py_result = fcDatevec(data)
    assert test_engine.equal(result, test_engine.convert(py_result))


# TODO: During migration remove this differential test
def test_fcDatevec_site_data_differential(test_engine):
    time_artifact_path = "tests/test_artifacts/cpdEvaluateUStarTh4Season20100901_artifacts/CA-Cbo_qca_ustar_2007/input_time_it_.csv"
    data = pd.read_csv(time_artifact_path, header=None).values.tolist()

    # Call the function
    expected_result = test_engine.fcDatevec(test_engine.convert(data), nargout=6)

    print("test_engine.fcDatevec(test_engine.convert(data), nargout=6) has run...\n\n")

    result = fcDatevec(data)

    assert test_engine.equal(result, expected_result)
