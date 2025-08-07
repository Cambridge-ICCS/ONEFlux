# Contains tests for testing the nlinfit function
# which performs nonlinear regression fitting, used in cpdAssignUStarTh.py
# for fitting annual sine curves to change point data

import pytest
import numpy as np
from oneflux_steps.ustar_cp_python.fcEqnAnnualSine import fcEqnAnnualSine


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
