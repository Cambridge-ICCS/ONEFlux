# Contains tests for testing the nlinfit function
# which performs nonlinear regression fitting, used in cpdAssignUStarTh.py
# for fitting annual sine curves to change point data

import pytest
import numpy as np
from oneflux_steps.ustar_cp_python.fcEqnAnnualSine import fcEqnAnnualSine
from hypothesis import given, strategies as st
from oneflux_steps.ustar_cp_python.utilities import nlinfit as python_nlinfit


@pytest.mark.parametrize(
    "xdata, ydata, expected",
    [
        ([1, 2, 3, 10], [4, 5, 6, 7], [-630.0498, 637.7461, -84.0255]),
        ([1, 2, 3, 10], [4, 5, 6, np.nan], [5.0, 58.1330, 2.0000]),
    ],
)
def test_nlinfit(test_engine, xdata, ydata, expected):
    """
    Test the nlinfit function with various inputs.
    """
    # Convert inputs to numpy arrays
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # Call the nlinfit function
    model_function = None
    # if the test engine is matlab then use fcEqnAnnualSine name
    if test_engine.language() == "matlab":
        model_function = "fcEqnAnnualSine"
    elif test_engine.language() == "python":
        model_function = fcEqnAnnualSine

    initial_guess = [1.0, 1.0, 1.0]
    result = test_engine.nlinfit(
        test_engine.convert(xdata),
        test_engine.convert(ydata),
        model_function,
        test_engine.convert(initial_guess),
    )

    # Check if the result matches the expected output
    assert np.allclose(result, expected, rtol=1e-4), (
        f"Expected {expected}, got {result}"
    )


#  Hypothesis differential test between matlab and python
# This test is to ensure that the nlinfit function behaves consistently across different engines
# # Generate a range of xdata and ydata values
@given(
    st.lists(st.floats(min_value=0, max_value=365), min_size=10, max_size=10),
    st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=10),
)
def test_nlinfit_differential(test_engine, xdata, ydata):
    """
    Differential test for nlinfit between MATLAB and Python engines.
    """
    # Convert inputs to numpy arrays
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # Define model function and initial guess
    initial_guess = [1.0, 1.0, 1.0]

    # Call nlinfit on MATLAB engine
    result_matlab = test_engine.nlinfit(
        test_engine.convert(xdata),
        test_engine.convert(ydata),
        "fcEqnAnnualSine",
        test_engine.convert(initial_guess),
    )

    # Call nlinfit on Python engine
    result_python = python_nlinfit(xdata, ydata, fcEqnAnnualSine, initial_guess)

    # Check if the results from both engines are close
    assert np.allclose(
        test_engine.unconvert(result_matlab), result_python, rtol=1e-4
    ), f"MATLAB result {result_matlab} differs from Python result {result_python}"


# FAILED tests/unit_tests/test_ustar_cp/test_nlinfit.py::test_nlinfit_differential
# - AssertionError: MATLAB result [[0.7877046890058979,6.771766110260001,6.777151682978612]]
# differs from Python result [0.01777358 1.01648187 1.01648029]
