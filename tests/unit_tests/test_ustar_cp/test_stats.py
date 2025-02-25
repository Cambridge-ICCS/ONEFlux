import pytest
import numpy as np

nan = np.nan

# Define the expected field names for StatsMT
expected_fields = [
    "n",
    "Cp",
    "Fmax",
    "p",
    "b0",
    "b1",
    "b2",
    "c2",
    "cib0",
    "cib1",
    "cic2",
    "mt",
    "ti",
    "tf",
    "ruStarVsT",
    "puStarVsT",
    "mT",
    "ciT",
]


# Test for the generate_statsMT function
def test_generate_statsMT(test_engine):
    # Generate the StatsMT struct
    StatsMT = test_engine.generate_statsMT()

    # Ensure all expected fields exist and are NaN
    for field in expected_fields:
        assert (
            field in StatsMT
        ), f"Missing field: {field}"  # Access fields like dict keys
        assert np.isnan(StatsMT[field]), f"Field {field} is not NaN"


stats_entry = {
    "n": nan,
    "Cp": nan,
    "Fmax": nan,
    "p": nan,
    "b0": nan,
    "b1": nan,
    "b2": nan,
    "c2": nan,
    "cib0": nan,
    "cib1": nan,
    "cic2": nan,
    "mt": nan,
    "ti": nan,
    "tf": nan,
    "ruStarVsT": nan,
    "puStarVsT": nan,
    "mT": nan,
    "ciT": nan,
}


# Test for the setup_Stats function
@pytest.mark.parametrize(
    "nBoot, nSeasons, nStrataX, expected_shape",
    [
        # Case 1: Basic 2x2x2 array
        (
            2,
            2,
            2,
            (
                [
                    [[stats_entry, stats_entry], [stats_entry, stats_entry]],
                    [[stats_entry, stats_entry], [stats_entry, stats_entry]],
                ]
            ),
        ),
        # TODO: check whether we need these tests
        # Case 2a: Single season, single strata, single boot
        # (1, 1, 1, stats_entry),
        # Case 3: No bootstrap iterations (nBoot=0)
        # (0, 2, 3, stats_entry),
    ],
)
def test_setup_Stats(test_engine, nBoot, nSeasons, nStrataX, expected_shape):
    # Call the MATLAB function
    Stats = test_engine.setup_Stats(nBoot, nSeasons, nStrataX, jsonencode=[0])

    assert test_engine.equal(Stats, expected_shape)
