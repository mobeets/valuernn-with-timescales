import numpy as np

def dprime_vector(Z1, Z2):
    """
    Compute per-feature d' between Z1 and Z2.
    
    Parameters
    ----------
    Z1 : ndarray of shape (T1, N)
        First set of samples (T1 trials, N features).
    Z2 : ndarray of shape (T2, N)
        Second set of samples (T2 trials, N features).
    
    Returns
    -------
    dprimes : ndarray of shape (N,)
        d' value for each feature.
    """
    m1 = Z1.mean(axis=0)
    m2 = Z2.mean(axis=0)
    v1 = Z1.var(axis=0, ddof=1)  # unbiased variance
    v2 = Z2.var(axis=0, ddof=1)
    
    # pooled std
    s = np.sqrt(0.5 * (v1 + v2))
    
    # avoid division by zero
    s[s == 0] = np.nan  # set zero std to NaN to avoid division by zero
    dprimes = (m1 - m2) / s
    dprimes[s == 0] = np.nan
    
    return dprimes
