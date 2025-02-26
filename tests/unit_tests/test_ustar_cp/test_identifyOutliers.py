import numpy as np
import pytest
from tests.conftest import flatten


@pytest.mark.parametrize(
    "input_data, threshold, expected_outlierFlag, expected_outlierIndices",
    [
        (
            np.linspace(0, 10, 11),
            5,
            [False, False, False, False, False, False, True, True, True, True, True],
            np.array([7, 8, 9, 10, 11]),
        ),
    ],
)
def test_identify_outliers(
    test_engine, input_data, threshold, expected_outlierFlag, expected_outlierIndices
):
    outlier_flag, outlie_Indices = test_engine.identifyOutliers(
        test_engine.convert(input_data), test_engine.convert(threshold), nargout=2
    )

    assert test_engine.equal(outlier_flag, test_engine.convert(expected_outlierFlag))
    assert test_engine.equal(
        outlie_Indices, test_engine.convert(expected_outlierIndices, index="to_python")
    )
