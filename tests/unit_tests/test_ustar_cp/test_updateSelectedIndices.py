import numpy as np
import pytest


@pytest.mark.parametrize(
    "iSelect, iOut, fSelect, fOut, expected_nSelect",
    [
        (
            [1, 2, 3, 4, 5],
            [2, 4],
            [True, True, False, True, False],
            [False, False, False, False, False],
            3,
        ),
        (
            [10, 20, 30, 40],
            [20],
            [True, False, True, True],
            [False, False, False, False],
            3,
        ),
        (
            [5, 10, 15, 20, 25],
            [],
            [True, True, True, True, True],
            [False, False, False, False, False],
            5,
        ),
        (
            [1, 3, 5, 7, 9],
            [1, 3, 9],
            [True, False, True, True, True],
            [False, True, False, False, False],
            2,
        ),
    ],
)
def test_update_selected_indices(
    test_engine, iSelect, iOut, fSelect, fOut, expected_nSelect
):
    """Test that updateSelectedIndices correctly removes outliers and updates selection."""

    updated_iSelect, updated_nSelect, updated_fSelect = (
        test_engine.updateSelectedIndices(
            test_engine.convert(iSelect),
            test_engine.convert(iOut),
            test_engine.convert(fSelect),
            test_engine.convert(fOut),
            nargout=3,
        )
    )

    # Check expected number of selections
    assert test_engine.equal(
        updated_nSelect, expected_nSelect
    ), f"Expected {expected_nSelect}, got {updated_nSelect}"
