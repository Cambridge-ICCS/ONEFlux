import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    # Create synthetic annual sine data
    # True parameters:
    true_amplitude = 2.0
    true_offset = 10.0
    true_phase = 50.0
    days = np.linspace(0, 365, 100)
    Cp = true_amplitude * np.sin(2 * np.pi * (days - true_phase) / 365.25) + true_offset
    # Add some noise
    noise = np.random.normal(0, 0.1, size=len(days))
    Cp_noisy = Cp + noise

    # We'll select all data points
    iSelect = np.arange(len(days))
    iSelect = iSelect[iSelect != 0]  # Cant index an array with zero in Matlab

    return days, Cp_noisy, iSelect, (true_offset, true_amplitude, true_phase)


def test_fit_output_shape_and_type(test_engine, synthetic_data):
    days, Cp_noisy, iSelect, true_sine = synthetic_data

    days = test_engine.convert(days)
    Cp_noisy = test_engine.convert(Cp_noisy)
    iSelect = test_engine.convert(iSelect)

    result = test_engine.fitAnnualSineCurve(days, Cp_noisy, iSelect)
    # Expecting [amplitude, offset, phase, r2]

    result = test_engine.convert(result)

    if len(result) == 1:
        result = result[0]

    assert test_engine.equal(len(result), 4)
    for val in result:
        assert isinstance(val, float)
    assert np.allclose(result[0], true_sine[0], rtol=0.1)  # test offset accuracy
    assert np.allclose(result[1], true_sine[1], rtol=0.1)  # test amplitude accuracy
    assert np.allclose(result[2], true_sine[2], rtol=0.1)  # test phase accuracy


def test_fit_on_synthetic_data(test_engine, synthetic_data):
    days, Cp_noisy, iSelect, (true_off, true_amp, true_ph) = synthetic_data

    result = test_engine.fitAnnualSineCurve(
        test_engine.convert(days),
        test_engine.convert(Cp_noisy),
        test_engine.convert(iSelect),
    )

    result = test_engine.convert(result)

    if len(result) == 1:  # Matlab case
        fitted_amp = result[0][1]
        fitted_off = result[0][0]
        fitted_ph = result[0][2]
        fitted_r2 = result[0][3]
    else:  # Python case
        fitted_amp = result[1]
        fitted_off = result[0]
        fitted_ph = result[2]
        fitted_r2 = result[3]

    # Check that fitted parameters are close to the true parameters.
    # Allow some tolerance due to noise.
    assert np.isclose(fitted_amp, true_amp, rtol=0.2)
    assert np.isclose(fitted_off, true_off, rtol=0.2)

    # Phase can wrap around, so it's trickier. We can check a mod difference:
    phase_diff = (fitted_ph - true_ph) % 365.25
    if phase_diff > 182.625:
        phase_diff = 365.25 - phase_diff
    assert phase_diff < 30  # Somewhat lenient

    # R-squared should be reasonably high for low-noise data
    assert fitted_r2 > 0.8


def test_constant_data(test_engine):
    # If the data is constant, the fit might fail or result in amplitude ~ 0
    days_array = np.linspace(0, 365, 100)
    days = test_engine.convert(days_array)
    Cp = test_engine.convert(np.ones_like(days_array) * 5.0)
    iSelect = np.arange(len(days_array))
    iSelect = iSelect[iSelect != 0]  # Cant index an array with zero in Matlab
    iSelect = test_engine.convert(iSelect)

    result = test_engine.fitAnnualSineCurve(days, Cp, iSelect)
    result = test_engine.convert(result)

    if len(result) == 1:
        fitted_amp = result[0][1]
        fitted_off = result[0][0]
        fitted_r2 = result[0][3]
    else:
        fitted_amp = result[1]
        fitted_off = result[0]
        fitted_r2 = result[3]

    assert abs(fitted_amp) < 0.5
    assert abs(fitted_off - 5.0) < 0.5
    # Phase doesn't matter much here, R^2 should be very high
    assert fitted_r2 > 0.9
