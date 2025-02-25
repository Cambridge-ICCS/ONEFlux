import numpy as np
import pytest
import matlab

@pytest.fixture
def sample_data():
    """Provides sample input data for testing the MATLAB function."""
    mt = np.linspace(0, 365, 100)  # Simulated time values in days
    Cp = 10 * np.sin(2 * np.pi * mt / 365) + np.random.normal(0, 0.5, 100)  # Noisy sine wave
    iSelect = np.ones_like(mt, dtype=bool)  # Select all points
    return mt, Cp, iSelect


def test_fit_annual_sine_curve(test_engine, sample_data):
    """Test that fitAnnualSineCurve executes correctly and returns expected values."""
    mt, Cp, iSelect = sample_data
    
    # Call MATLAB function
    result = test_engine.fitAnnualSineCurve(mt, Cp, iSelect, nargout=1)

    # Sine wave properties
    sine_offset = result[0][0]
    sine_amp = result[0][1]
    sine_phase = result[0][2]
    
    # Validate the R-squared value
    r2_value = result[0][3]

    assert -0.3 < sine_offset < 0.3
    assert 9.5 < sine_amp < 10.5
    assert 363 < sine_phase or sine_phase < 1
    assert r2_value > 0.99

def test_fit_annual_sine_curve_partial_selection(test_engine, sample_data):
    """Test the function with only part of the dataset selected."""
    mt, Cp, iSelect = sample_data
    iSelect[:50] = False  # Only use the last 50 data points
    

    result = test_engine.fitAnnualSineCurve(mt, Cp, iSelect, nargout=1)

    # Sine wave properties
    sine_offset = result[0][0]
    sine_amp = result[0][1]
    sine_phase = result[0][2]
    
    # Validate the R-squared value
    r2_value = result[0][3]
    
    assert -0.3 < sine_offset < 0.3
    assert 9.5 < sine_amp < 10.5
    assert sine_phase < 1 or sine_phase > 363
    assert r2_value > 0.96