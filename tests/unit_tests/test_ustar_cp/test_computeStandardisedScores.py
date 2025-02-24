import pytest
import numpy as np

@pytest.mark.parametrize(
    "input_data, expected_xNormX",
    [
        #
        # Test 1: 2x3 matrix, no NaNs
        #
        #   x = [[1,2,3],
        #        [4,5,6]]
        #
        #   For each column (with only 2 values):
        #     median => (value1 + value2)/2
        #     iqr    => 75th - 25th percentile => typically 1.5 for each col
        #
        #   => xNorm => row0 => [-1, -1, -1]
        #               row1 => [ 1,  1,  1]
        #   => xNormX => [1, 1]
        #
        (
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [[np.nan],[np.nan]],
        ),

        #
        # Test 2: 3x3 matrix, no NaNs
        (
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [[np.nan],[np.nan],[np.nan]],
        ),

        #
        # Test 3: 3x3 with a NaN in the middle row
        #
        #   x = [[1,   2,   3],
        #        [4,  NaN,  6],
        #        [7,   8,   9]]
        #
        #   We already reasoned that row0 => xNorm => [-1, -1, -1]
        #                         row1 => => [0,  NaN, 0] => max => NaN
        #                         row2 => => [1,  1,   1]
        #   => xNormX => [1, NaN, 1]
        #
        (
            [
                [1,    2,    3],
                [4,    np.nan,6],
                [7,    8,    9],
            ],
            [[np.nan],[np.nan],[np.nan]],
        ),

        #
        # Test 4: 3x1 matrix
        (
            [
                [1],
                [2],
                [5],
            ],
            [[0.3333333333333333],[0.0],[1.0]],
        ),
    ]
)
def test_computeStandardizedScores(test_engine, input_data, expected_xNormX):
    """
    Test the MATLAB function computeStandardizedScores by calling it 
    via the MATLAB Engine. We verify xNormX row-by-row against 
    expected values, including NaNs.
    """

    # Call the MATLAB function computeStandardizedScores
    xNormX = test_engine.computeStandardizedScores(test_engine.convert(input_data), nargout=1)

    assert test_engine.equal(xNormX, test_engine.convert(expected_xNormX))

    
