import pytest
import numpy as np

@pytest.mark.parametrize("input_data, expected_rows, expected_columns", [
    (np.random.randn(100, 5), 100, 1),
    (100*np.ones((100, 5)), 100, 1),
])

def test_compute_standardized_scores(test_engine, input_data, expected_rows, expected_columns):
    """Test that computeStandardizedScores executes correctly and returns expected values."""
    
    # Call MATLAB function
    result = test_engine.computeStandardizedScores(input_data, nargout=1)

    len_result = len(test_engine.convert(result))

    assert test_engine.equal(len_result, test_engine.convert(expected_rows))
    