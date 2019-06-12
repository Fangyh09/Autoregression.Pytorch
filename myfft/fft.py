from myfft.fft_instance import get_fft_instance
from myfft.fft_filters import get_fft_filter
import numpy as np

def fft_recompose(R, mu, sigma):
    """Recompose a cascade by inverting the normalization and summing the
    cascade levels.
    
    Parameters
    ----------
    R : array_like
      
    """
    # R_rc = [(R[i, :, :] * sigma[i]) + mu[i] for i in range(len(mu))]
    # R_rc = np.sum(np.stack(R_rc), axis=0)
    R_rc = np.sum(np.stack(R), axis=0)

    return R_rc

def fft_decompose(R, ar_order=2, n_cascade_levels=8, R_thr=-10):    
    """
    R: [3xhxw]
    R_thr: for position whose value < R_thr is masked.
    """
    # R: [3xhxw]
    # filter_method = get_fft_filter(method = "guass")
    M = R.shape[1]
    N = R.shape[2]

    MASK_thr = np.logical_and.reduce([R[i, :, :] >= R_thr for i in range(R.shape[0])])

    filters = get_fft_filter((M, N), n_cascade_levels, method="gaussian")
    fft_instance = get_fft_instance("numpy", (M, N))

    R_d = []
    for i in range(ar_order + 1):
        R_ = decomposition_fft(R[i, :, :], filters, fft_method=fft_instance, MASK=MASK_thr)
        R_d.append(R_)
    return R_d


def decomposition_fft(X, filter, fft_method, MASK=None):
    """Decompose a 2d input field into multiple spatial scales by using the Fast
    Fourier Transform (FFT) and a bandpass filter.

    Parameters
    ----------
    X : array_like
        Two-dimensional array containing the input field. All values are
        required to be finite.
    filter : dict
        A filter returned by a method implemented in
        :py:mod:`pysteps.cascade.bandpass_filters`.

    Other Parameters
    ----------------
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".
    MASK : array_like
        Optional mask to use for computing the statistics for the cascade
        levels. Pixels with MASK==False are excluded from the computations.

    Returns
    -------
    out : ndarray
        A dictionary described in the module documentation.
        The number of cascade levels is determined from the filter
        (see :py:mod:`pysteps.cascade.bandpass_filters`).

    """
    # fft_method = kwargs.get("fft_method", "numpy")
    # if type(fft_method) == str:
    #     # fft = get_fft_filter(fft, shape=X.shape)
    #     fft = get_fft_instance(fft_method, shape=X.shape)
    # else:
    fft = fft_method
    

    # MASK = kwargs.get("MASK", None)
    ### chech here ###
    if len(X.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    if MASK is not None and MASK.shape != X.shape:
        raise ValueError("Dimension mismatch between X and MASK:"
                         + "X.shape=" + str(X.shape)
                         + ",MASK.shape" + str(MASK.shape))

    if X.shape[0] != filter["weights_2d"].shape[1]:
        raise ValueError(
            "dimension mismatch between X and filter: "
            + "X.shape[0]=%d , " % X.shape[0]
            + "filter['weights_2d'].shape[1]"
              "=%d" % filter["weights_2d"].shape[1])

    if int(X.shape[1] / 2) + 1 != filter["weights_2d"].shape[2]:
        raise ValueError(
            "Dimension mismatch between X and filter: "
            "int(X.shape[1]/2)+1=%d , " % (int(X.shape[1] / 2) + 1)
            + "filter['weights_2d'].shape[2]"
              "=%d" % filter["weights_2d"].shape[2])

    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    ### End of check ###


    result = {}
    means = []
    stds = []

    F = fft.rfft2(X)
    X_decomp = []
    for k in range(len(filter["weights_1d"])):
        W_k = filter["weights_2d"][k, :, :]
        X_ = fft.irfft2(F * W_k)
        # from boxx import loga
        # loga(F)
        # loga(W_k)
        X_decomp.append(X_)

        if MASK is not None:
            X_ = X_[MASK]
        means.append(np.mean(X_))
        stds.append(np.std(X_))

    result["cascade_levels"] = np.stack(X_decomp)
    result["means"] = means
    result["stds"] = stds

    return result

