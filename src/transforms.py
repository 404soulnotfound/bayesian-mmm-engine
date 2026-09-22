"""
Differentiable Tensor Transformations for Adstock Decay and Hill Saturation.
Compatible with PyTensor and PyMC NUTS sampler.
"""

import pytensor.tensor as pt
import numpy as np


def geometric_adstock_tensor(x, alpha):
    """
    Computes geometric adstock decay using pytensor recurrence (scan).
    x: 1D or 2D tensor of media spends over time.
    alpha: decay rate scalar in (0, 1).
    """
    def step(x_t, prev_adstock, alpha):
        return x_t + alpha * prev_adstock

    adstock_seq, _ = pt.scan(
        fn=step,
        sequences=[x],
        outputs_info=[pt.zeros_like(x[0])],
        non_sequences=[alpha],
        strict=True
    )
    return adstock_seq


def hill_saturation_tensor(x, K, S):
    """
    Differentiable Hill saturation transformation:
    Hill(x, K, S) = (x^S) / (K^S + x^S)
    x: adstocked spend tensor
    K: half-saturation point (> 0)
    S: slope/shape parameter (> 0)
    """
    x_safe = pt.maximum(x, 1e-6)
    x_pow = pt.power(x_safe, S)
    k_pow = pt.power(K, S)
    return x_pow / (k_pow + k_pow * 0.0 + x_pow)


# Fast vectorized NumPy versions for real-time app simulation & optimization
def fast_geometric_adstock_np(spend_matrix: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Vectorized NumPy geometric adstock for array shape (n_time, n_channels).
    spend_matrix: (T, K)
    alphas: (K,)
    """
    T, K = spend_matrix.shape
    adstocked = np.zeros_like(spend_matrix, dtype=float)
    for t in range(T):
        if t == 0:
            adstocked[t] = spend_matrix[t]
        else:
            adstocked[t] = spend_matrix[t] + alphas * adstocked[t - 1]
    return adstocked


def fast_hill_saturation_np(x: np.ndarray, K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Vectorized NumPy Hill saturation for array shape (T, K).
    """
    x_safe = np.maximum(x, 1e-6)
    x_pow = np.power(x_safe, S)
    k_pow = np.power(K, S)
    return x_pow / (k_pow + x_pow)
