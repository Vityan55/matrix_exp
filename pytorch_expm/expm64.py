# expm64.py
from __future__ import division, print_function, absolute_import  # для совместимости с Python 2/3

import math
import numpy as np
import torch
import scipy.special  # для вычисления биномиальных коэффициентов

# -------------------------
# Helpers (вспомогательные функции)
# -------------------------

def _onenorm(A):
    """
    Возвращает 1-норму матрицы A (максимальная сумма абсолютных значений по столбцам)
    """
    return torch.norm(A, 1).item()

def _onenorm_matrix_power_nnm(A, p):
    """
    Приближенная оценка нормы A^p для целого p >= 0
    Используется для безопасного выбора масштаба в Pade13.
    """
    if int(p) != p or p < 0:
        raise ValueError('expected non-negative integer p')
    p = int(p)

    v = torch.ones((A.shape[0], 1), dtype=A.dtype, device=A.device)  # вектор единиц
    M = A.t()  # транспонируем для быстрого умножения
    for _ in range(p):
        v = M.mm(v)  # перемножаем p раз
    return torch.max(v).item()  # оценка нормы

def _ident_like(A):
    """
    Возвращает единичную матрицу того же размера, что и A
    """
    return torch.eye(A.shape[0], A.shape[1], dtype=A.dtype, device=A.device)

# -------------------------
# Pade(13) helper (только для Pade 13)
# -------------------------

class _ExpmPadeHelper:
    """
    Вспомогательный класс для вычисления Pade(13)
    с кешированием степеней A^2, A^4, A^6
    """
    def __init__(self, A):
        self.A = A
        self.ident = _ident_like(A)  # единичная матрица того же размера

        self._A2 = None
        self._A4 = None
        self._A6 = None

    @property
    def A2(self):
        """
        Ленивое вычисление A^2
        """
        if self._A2 is None:
            self._A2 = self.A @ self.A
        return self._A2

    @property
    def A4(self):
        """
        Ленивое вычисление A^4 через A^2
        """
        if self._A4 is None:
            self._A4 = self.A2 @ self.A2
        return self._A4

    @property
    def A6(self):
        """
        Ленивое вычисление A^6 через A^4 и A^2
        """
        if self._A6 is None:
            self._A6 = self.A4 @ self.A2
        return self._A6

    def pade13_scaled(self, s):
        """
        Вычисляет U и V для Pade13 с масштабированием 2^-s
        """
        # коэффициенты Pade(13)
        b = (
            64764752532480000., 32382376266240000., 7771770303897600.,
            1187353796428800., 129060195264000., 10559470521600.,
            670442572800., 33522128640., 1323241920., 40840800.,
            960960., 16380., 182., 1.
        )

        # масштабируем матрицу
        B  = self.A  * 2**(-s)
        B2 = self.A2 * 2**(-2*s)
        B4 = self.A4 * 2**(-4*s)
        B6 = self.A6 * 2**(-6*s)

        # вычисляем U и V для Pade аппроксимации
        U2 = B6 @ (b[13]*B6 + b[11]*B4 + b[9]*B2)
        U  = B  @ (U2 + b[7]*B6 + b[5]*B4 + b[3]*B2 + b[1]*self.ident)

        V2 = B6 @ (b[12]*B6 + b[10]*B4 + b[8]*B2)
        V  = V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*self.ident

        return U, V

# -------------------------
# Core algorithm
# -------------------------

def _solve_P_Q(U, V):
    """
    Решает (V-U)X = V+U для Pade
    """
    return torch.linalg.solve(U + V, -U + V)

def _ell(A, m):
    """
    Вычисление дополнительного масштаба для метода Higham
    """
    p = 2*m + 1
    choose_2p_p = scipy.special.comb(2*p, p, exact=True)  # биномиальный коэффициент
    abs_c_recip = float(choose_2p_p * math.factorial(2*p + 1))

    u = 2.0 ** -53  # машинный эпсилон для float64

    A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), p)
    if not A_abs_onenorm:
        return 0

    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    return max(int(np.ceil(np.log2(alpha / u) / (2 * m))), 0)

# -------------------------
# Public API
# -------------------------

def expm64(A):
    """
    Вычисление exp(A) для float64 с использованием Pade13 и масштабирования
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # тривиальный случай для 1x1
    if A.shape == (1, 1):
        return torch.exp(A)

    # ранний выход для nilpotent матриц (A^2 ≈ 0)
    if torch.norm(A @ A) < 1e-12:
        I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        return I + A

    h = _ExpmPadeHelper(A)  # вспомогательный объект с кешем степеней

    theta_13 = 4.25  # рекомендуемое значение для Pade13

    # безопасная оценка масштаба
    eta = max(
        _onenorm(h.A4)**0.25 if _onenorm(h.A4) > 0 else 0,
        _onenorm(h.A6)**(1/6) if _onenorm(h.A6) > 0 else 0
    )

    s = 0
    if eta > 0:
        s = max(int(np.ceil(np.log2(eta / theta_13))), 0)

    s += _ell(2**(-s) * h.A, 13)  # добавляем дополнительный масштаб

    # вычисляем U и V для Pade13 с масштабированием
    U, V = h.pade13_scaled(s)
    X = _solve_P_Q(U, V)  # решаем систему для экспоненты

    return torch.matrix_power(X, 2**s)  # возводим в 2^s для восстановления исходного масштаба