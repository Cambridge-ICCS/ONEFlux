import numpy as np
import pytest
from tests.conftest import flatten

@pytest.mark.parametrize("input_data, threshold, expected_outlierFlag, expected_outlierIndices", [
    (np.linspace(0, 10, 11), 5, [False,False,False,False,False,False,True,True,True,True,True], np.array([5,6,7,8,9])),
])

def test_identify_outliers(test_engine, input_data, threshold, expected_outlierFlag, expected_outlierIndices):

    result = test_engine.identifyOutliers(input_data, threshold)

    assert result[0] == test_engine.convert(expected_outlierFlag)