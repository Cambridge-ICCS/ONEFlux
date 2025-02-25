import pytest
import numpy as np


@pytest.mark.parametrize(
    (
        "xCp",  # 2D or 3D matrix to pass into MATLAB
        "iSelect",  # numeric (1-based) indices into xCp
        "nDim",
        "nWindows",
        "nStrata",
        "nBoot",
        "expected_CpA",
        "expected_nA",
        "expected_xCpSelect",
    ),
    [
        #
        # Test Case 1: 2D, shape(2,3).
        #   In MATLAB, xCp(1,1)=1, xCp(2,1)=4, xCp(1,2)=2, xCp(2,2)=5, xCp(1,3)=3, xCp(2,3)=6
        #   We select iSelect=[2,4,6], i.e. xCp(2)=4, xCp(4)=5, xCp(6)=6.
        #   Then:
        #     xCpSelect => [NaN,4, NaN,5, NaN,6] in linear indexing => shape(2,3):
        #       [[NaN, NaN, NaN],
        #        [ 4 ,  5 ,  6 ]]
        #   CpA => nanmean(xCpSelect) => row-wise mean => [4,5,6]  (a 1x3 vector in MATLAB)
        #   nA  => sum(~isnan(xCpSelect)) => [1,1,1]
        #
        # NOTE: We'll store "expected_xCpSelect" as a nested Python list to compare shape(2,3).
        #
        (
            # xCp as a nested list to form 2D [2,3]
            [[1, 2, 3], [4, 5, 6]],
            [2, 4, 6],  # 1-based indices => picks xCp(2)=4, xCp(4)=5, xCp(6)=6
            2,  # nDim
            2,  # nWindows
            1,  # nStrata
            3,  # nBoot
            [4.0, 5.0, 6.0],  # expected_CpA
            [1, 1, 1],  # expected_nA
            [[np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]],
        ),
        #
        # Test Case 2: 3D, shape(2,2,2).
        #   In MATLAB (column-major), xCp(1,1,1)=1, xCp(2,1,1)=2, xCp(1,2,1)=3, xCp(2,2,1)=4,
        #                         xCp(1,1,2)=5, xCp(2,1,2)=6, xCp(1,2,2)=7, xCp(2,2,2)=8
        #   iSelect=[1,8] => picks xCp(1)=1, xCp(8)=8.
        #   => xCpSelect => [1, NaN, NaN, NaN, NaN, NaN, NaN, 8] => shape(2,2,2) => in "layer" form:
        #
        #   layer 1 => [[1,   NaN],
        #               [NaN, NaN]]
        #   layer 2 => [[NaN, NaN],
        #               [NaN, 8]]
        #
        #   Then we do nDim=3 => reshape => (nWindows*nStrata, nBoot) => (2*2, 2) => (4,2).
        #   We get a 4x2 array => only [1,8] are non-NaN => so:
        #       CpA => [1, 8]  (i.e. nanmean along dimension=1 => a row vector 1x2)
        #       nA  => [1, 1]  (count of non-NaN in each column)
        (
            # xCp as nested lists for shape(2,2,2). The dimension order in Python is [row, col, "page"].
            # We'll pass it as we want MATLAB to interpret it with column-major flattening.
            [
                [[1, 5], [3, 7]],  # "row 1"
                [[2, 6], [4, 8]],  # "row 2"
            ],
            [1, 8],  # picks xCp(1)=1, xCp(8)=8 in MATLAB
            3,  # nDim
            2,  # nWindows
            2,  # nStrata
            2,  # nBoot
            [1.0, 8.0],  # expected_CpA (1x2 row vector)
            [1, 1],  # expected_nA
            [[[1.0, np.nan], [np.nan, np.nan]], [[np.nan, np.nan], [np.nan, 8.0]]],
        ),
    ],
)
def test_aggregateSeasonalAndAnnualValues(
    test_engine,
    xCp,
    iSelect,
    nDim,
    nWindows,
    nStrata,
    nBoot,
    expected_CpA,
    expected_nA,
    expected_xCpSelect,
):
    """
    Test the function aggregateSeasonalAndAnnualValues

    Explanation:
    ------------
    - xCp is provided in a shape that should be [nWindows, nBoot] if nDim=2,
      or [nWindows, nStrata, nBoot] if nDim=3. Because MATLAB uses column-major order,
      the exact flattening can be tricky in Python. We keep the nested structure so
      the final shape is correct in MATLAB once passed through the engine.
    - iSelect is 1-based, matching MATLAB indexing. For instance, iSelect=[1,8] means
      xCp(1) and xCp(8) are selected.
    - The function returns three outputs: [CpA, nA, xCpSelect].
      We check each against the expected result (accounting for possible shape and
      dimensional differences).
    """

    CpA, nA, xCpSelect = test_engine.aggregateSeasonalAndAnnualValues(
        test_engine.convert(xCp),
        test_engine.convert(iSelect, index="to_python"),
        test_engine.convert(nDim),
        test_engine.convert(nWindows),
        test_engine.convert(nStrata),
        test_engine.convert(nBoot),
        nargout=3,
    )

    assert test_engine.equal(CpA, test_engine.convert(expected_CpA))
    assert test_engine.equal(nA, test_engine.convert(expected_nA))
    assert test_engine.equal(xCpSelect, test_engine.convert(expected_xCpSelect))
