# expm_pade.py
import torch  # для работы с тензорами и матрицами
import math   # для математических операций, log2, ceil
from expm64 import expm64  # fallback метод для больших порядков (Higham)

# ============================
# Вспомогательные функции
# ============================

def _onenorm(A):
    """
    Вычисляет 1-норму матрицы A (максимум по сумме модулей столбцов)
    Возвращает скаляр
    """
    return torch.norm(A, 1).item()


def _scale_matrix(A, s):
    """
    Масштабирует матрицу A на 2^-s
    Используется для метода масштабирования и возведения в степень (scaling & squaring)
    """
    return A * (2.0 ** -s)


def _squaring(X, s):
    """
    Возводит матрицу X в степень 2^s через s последовательных умножений
    Используется после масштабирования для восстановления исходного экспоненциального значения
    """
    for _ in range(s):
        X = X @ X  # матричное умножение
    return X


# ============================
# Реализации рациональных аппроксимаций Паде
# ============================

def _pade3(A):
    """
    Аппроксимация exp(A) с помощью Паде степени 3
    """
    n = A.size(0)
    I = torch.eye(n, device=A.device, dtype=A.dtype)  # единичная матрица

    A2 = A @ A  # A^2

    b = [120., 60., 12., 1.]  # коэффициенты Паде

    U = A @ (b[3]*A2 + b[1]*I)  # числитель
    V = b[2]*A2 + b[0]*I        # знаменатель

    # решение системы (V - U) X = (V + U)
    return torch.linalg.solve(V - U, V + U)


def _pade5(A):
    """
    Аппроксимация exp(A) с помощью Паде степени 5
    """
    n = A.size(0)
    I = torch.eye(n, device=A.device, dtype=A.dtype)

    A2 = A @ A
    A4 = A2 @ A2

    b = [30240., 15120., 3360., 420., 30., 1.]

    U = A @ (b[5]*A4 + b[3]*A2 + b[1]*I)
    V = b[4]*A4 + b[2]*A2 + b[0]*I

    return torch.linalg.solve(V - U, V + U)


def _pade7(A):
    """
    Аппроксимация exp(A) с помощью Паде степени 7
    """
    n = A.size(0)
    I = torch.eye(n, device=A.device, dtype=A.dtype)

    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2

    b = [17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.]

    U = A @ (b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I

    return torch.linalg.solve(V - U, V + U)


def _pade9(A):
    """
    Аппроксимация exp(A) с помощью Паде степени 9
    """
    n = A.size(0)
    I = torch.eye(n, device=A.device, dtype=A.dtype)

    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    A8 = A4 @ A4

    b = [17643225600., 8821612800., 2075673600., 302702400.,
         30270240., 2162160., 110880., 3960., 90., 1.]

    U = A @ (b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I

    return torch.linalg.solve(V - U, V + U)


# ============================
# Основная функция expm_pade
# ============================

def expm_pade(A, order=13):
    """
    Вычисляет exp(A) с помощью рациональной аппроксимации Паде
    и метода scaling & squaring.

    Параметры:
        A : [n,n] матрица (float64 предпочтительно)
        order : степень Паде (3,5,7,9)
    """
    # приводим к double, если нужно
    if A.dtype != torch.float64:
        A = A.double()

    normA = _onenorm(A)  # 1-норма матрицы

    # пороговые значения theta по Higham
    theta = {
        3: 1.495585217958292e-2,
        5: 2.539398330063230e-1,
        7: 9.504178996162932e-1,
        9: 2.097847961257068,
        13: 5.371920351148152
    }

    # выбор степени Паде
    if order == 3:
        m = 3
    elif order == 5:
        m = 5
    elif order == 7:
        m = 7
    elif order == 9:
        m = 9
    else:
        return expm64(A)  # fallback на Higham

    theta_m = theta[m]

    # ============================
    # Масштабирование (scaling)
    # ============================
    if normA <= theta_m:
        s = 0  # не масштабируем
    else:
        # вычисляем степень масштабирования s = ceil(log2(||A||/theta_m))
        s = int(math.ceil(math.log2(normA / theta_m)))

    A_scaled = _scale_matrix(A, s)  # делим матрицу на 2^s

    # ============================
    # Вычисление Паде
    # ============================
    if m == 3:
        X = _pade3(A_scaled)
    elif m == 5:
        X = _pade5(A_scaled)
    elif m == 7:
        X = _pade7(A_scaled)
    elif m == 9:
        X = _pade9(A_scaled)
    else:
        return expm64(A)

    # ============================
    # Возведение в степень 2^s (squaring)
    # ============================
    X = _squaring(X, s)

    return X