import numpy as np

def interpolate_coefficients(array1, array2, lam):
    """
    Interpolate between two sets of Fourier coefficients for a 3D surface.
    
    Parameters
    ----------
    array1 : ndarray
        First array of Fourier coefficients
    array2 : ndarray
        Second array of Fourier coefficients
    lam : float
        Interpolation parameter, typically between 0 and 1.
    
    Returns
    -------
    interpolated_array : ndarray
        Interpolated Fourier coefficients (shape: (121,))
    """

    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    interpolated_array = (1 - lam) * array1 + lam * array2
    
    return np.array(interpolated_array)