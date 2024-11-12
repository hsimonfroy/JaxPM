import jax.numpy as jnp
import numpy as np


def fftk(shape, symmetric=True, finite=False, dtype=np.float32):
    """ 
    Return wave-vectors for a given shape
    """
    k = []
    for d in range(len(shape)):
        kd = np.fft.fftfreq(shape[d])
        kd *= 2 * np.pi
        kdshape = np.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) - 1:
            kd = kd[:shape[d] // 2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)

        k.append(kd.astype(dtype))
    del kd, kdshape
    return k


def gradient_kernel(kvec, direction, order=1):
    """
    Computes the gradient kernel in the requested direction
    
    Parameters
    -----------
    kvec: list
        List of wave-vectors in Fourier space
    direction: int
        Index of the direction in which to take the gradient

    Returns
    --------
    wts: array
        Complex kernel values
    """
    ki = kvec[direction]
    if order == 0:
        pass
    elif order ==1:
        ki = (8. * np.sin(ki) - np.sin(2 * ki)) / 6.
    return 1j * ki


def invlaplace_kernel(kvec, order=1):
    """
    Compute the inverse Laplace kernel.

    cf. [Feng+2016](https://arxiv.org/pdf/1603.00476)

    Parameters
    -----------
    kvec: list
        List of wave-vectors

    Returns
    --------
    wts: array
        Complex kernel values
    """
    if order == 0:
        kk = sum(ki**2 for ki in kvec)
    elif order == 1:
        kk = sum((ki * np.sinc(ki / (2 * np.pi)))**2 for ki in kvec)
    kk_nozeros = np.where(kk==0, 1, kk) 
    return - np.where(kk==0, 0, 1 / kk_nozeros)


def longrange_kernel(kvec, r_split):
    """
    Computes a long range kernel

    Parameters
    -----------
    kvec: list
        List of wave-vectors
    r_split: float
        Splitting radius
        
    Returns
    --------
    wts: array
        Complex kernel values
    
    TODO: @modichirag add documentation
    """
    if r_split != 0:
        kk = sum(ki**2 for ki in kvec)
        return np.exp(-kk * r_split**2)
    else:
        return 1.


def cic_compensation(kvec):
    """
    Computes cic compensation kernel.
    Adapted from https://github.com/bccp/nbodykit/blob/a387cf429d8cb4a07bb19e3b4325ffdf279a131e/nbodykit/source/mesh/catalog.py#L499
    Itself based on equation 18 (with p=2) of
          [Jing et al 2005](https://arxiv.org/abs/astro-ph/0409240)

    Parameters:
    -----------
    kvec: list
        List of wave-vectors
        
    Returns:
    --------
    wts: array
        Complex kernel values
    """
    wts = 1
    for ki in kvec:
        wts = wts * np.sinc(ki / (2 * np.pi))
    return wts**-2


def PGD_kernel(kvec, kl, ks):
    """
    Computes the PGD kernel

    Parameters:
    -----------
    kvec: list
        List of wave-vectors
    kl: float
        Initial long range scale parameter
    ks: float
        Initial dhort range scale parameter

    Returns:
    --------
    v: array
        Complex kernel values
    """
    kk = sum(ki**2 for ki in kvec)
    kl2 = kl**2
    ks4 = ks**4
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    v = jnp.exp(-kl2 / kk) * jnp.exp(-kk**2 / ks4)
    imask = (~(kk == 0)).astype(int)
    v *= imask
    return v
