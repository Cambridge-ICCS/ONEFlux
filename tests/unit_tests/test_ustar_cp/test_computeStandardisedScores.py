import pytest
import numpy as np
from tests.conftest import parse_testcase, validate_against_site_data


@pytest.mark.parametrize(
    "input_data, expected_xNormX",
    [
        #
        # Test 1: 3x2 matrix, no NaNs
        (
            [[1, 4], [2, 5], [3, 6]],
            # 1x2
            [[np.nan, np.nan]],
        ),
        # Test 2: 3x3 matrix, no NaNs
        (
            [
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
            ],
            # 3x1 result
            [[np.nan, np.nan, np.nan]],
        ),
        #
        # Test 3: 3x3 with a NaN in the middle column
        (
            [
                [1, 4, 7],
                [2, np.nan, 8],
                [3, 6, 9],
            ],
            [[np.nan, np.nan, np.nan]],
        ),
        #
        # Test 4: 1x3 matrix
        (
            [
                [1, 2, 5],
            ],
            [[0.3333333333333333, 0.0, 1.0]],
        ),
    ],
)
def test_computeStandardizedScores(test_engine, input_data, expected_xNormX):
    """
    Test the function computeStandardizedScores by calling it
    via the 'test engine'. We verify xNormX row-by-row against
    expected values, including NaNs.
    """

    input_data = np.array(input_data)
    expected_xNormX = np.array(expected_xNormX)
    xNormX = test_engine.computeStandardizedScores(input_data, nargout=1)

    assert test_engine.equal(xNormX, expected_xNormX)


def test_against_testcases(test_engine):
    input_names = ["regressionMatrix"]
    output_names = ["standardizedScores"]
    artifacts_dir = "tests/test_artifacts/computeStandardizedScores_artifacts/"

    def inner_function(input_data):
        return test_engine.computeStandardizedScores(input_data["regressionMatrix"])

    validate_against_site_data(
        test_engine,
        "computeStandardizedScores",
        input_names,
        output_names,
        artifacts_dir,
        inner_function,
    )
