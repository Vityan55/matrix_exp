# expm_sketch.py
import math
import torch
from expm_pade import expm_pade


def gaussian_sketch(n, k, device, dtype, generator=None):
    """Гауссовская матрица S (n x k) с масштабом 1/sqrt(k), E[S S^T] ≈ I."""
    S = torch.randn(n, k, device=device, dtype=dtype, generator=generator)
    S = S / (k ** 0.5)
    return S


def orthonormal_sketch(n, k, device, dtype, generator=None):
    """
    Случайное ортонормированное подпространство: Q ∈ R^{n×k}, Q^T Q = I.
    Стабильнее, чем «голый» гауссовский S, для сжатия B = Q^T A Q.
    """
    G = torch.randn(n, k, device=device, dtype=dtype, generator=generator)
    Q, _ = torch.linalg.qr(G, mode="reduced")
    return Q


def _effective_sketch_dim(n, k_user, k_cap=512):
    """
    Для больших n фиксированное k даёт огромную ошибку: поднимаем k сублинейно по n.
    k_cap ограничивает стоимость O(n^2 k) и память.
    """
    if n <= 1:
        return 0

    if k_user >= n:
        return n

    k_boost = max(
        k_user,
        min(k_cap, int(math.ceil(4.0 * math.sqrt(float(n)))))
    )

    return min(k_boost, n - 1)


def expm_pade_orthogonal_sketch(A, params):
    """
    Паде + скетчинг в случайном подпространстве (ортогональная проекция).

    Идея:
        B = Q^T A Q
        exp(A) ≈ Q exp(B) Q^T (+ усреднение по нескольким Q)

    params:
        (order, k)
        (order, k, r)
        (order, k, r, k_cap)
    """

    n = A.size(0)

    if n <= 1:
        return expm_pade(A, params[0])

    order = int(params[0])
    k_user = int(params[1])
    r = int(params[2]) if len(params) > 2 else None
    k_cap = int(params[3]) if len(params) > 3 else 512

    k_eff = _effective_sketch_dim(n, k_user, k_cap=k_cap)

    if k_eff >= n:
        return expm_pade(A, order)

    if r is None:
        r = 2 if n > 128 else 1

    device, dtype = A.device, A.dtype
    acc = torch.zeros((n, n), device=device, dtype=dtype)

    for _ in range(r):
        Q = orthonormal_sketch(n, k_eff, device, dtype)
        AQ = A @ Q
        B = Q.T @ AQ
        X_small = expm_pade(B, order)
        acc = acc + Q @ X_small @ Q.T

    return acc / r


def expm_pade_sketch(A, params):
    """Совместимость: старый вход (order, k)."""
    return expm_pade_orthogonal_sketch(A, params)