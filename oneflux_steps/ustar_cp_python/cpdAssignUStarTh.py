import numpy as np
def identifyOutliers(x_norm_x, threshold):
    """
    Identifies outliers based on standardized scores.

    Parameters:
    x_norm_x (numpy.ndarray): Array of normalized values.
    threshold (float): Threshold value for outlier detection.

    Returns:
    tuple: A boolean array (f_out) indicating outliers and an array (i_out) of outlier indices.
    """
    f_out = (x_norm_x > threshold)  # Ensures f_out is a boolean array
    i_out = np.where(f_out)[0]  # Indices of outliers

    
    return f_out, i_out  # Ensure f_out remains a boolean array