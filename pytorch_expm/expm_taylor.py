# expm_taylor.py
import torch  # для работы с матрицами и тензорами
import math  # для математических операций, например sqrt, log2

# ============================
# Настройки степеней Тейлора
# ============================
# Возможные степени разложения Тейлора
degs = [1, 2, 4, 8, 12, 18]

# theta values для float32 (single precision)
# пороговые значения нормы для выбора степени разложения
thetas_single = [
    1.192092800768788e-07,  # для m_vals = 1
    5.978858893805233e-04,  # для m_vals = 2
    5.116619363445086e-02,  # для m_vals = 4
    5.800524627688768e-01,  # для m_vals = 8
    1.461661507209034e+00,  # для m_vals = 12
    3.010066362817634e+00  # для m_vals = 18
]


# ============================
# Функция быстрого возведения матрицы в степень 2^k для батча
# ============================
def matrix_power_two_batch(A, k):
    """
    Возводит батч матриц A в степень 2^k по каждому элементу батча.
    Оптимизация для случаев, когда нужно делать масштабирование.
    """
    orig_size = A.size()  # сохраняем исходную размерность
    A, k = A.flatten(0, -3), k.flatten()  # превращаем в "плоский" батч
    ksorted, idx = torch.sort(k)  # сортируем степени
    count = torch.bincount(ksorted)  # считаем количество каждой степени
    nonzero = torch.nonzero(count, as_tuple=False)  # индексы ненулевых степеней
    A = torch.matrix_power(A, 2 ** ksorted[0])  # первая степень
    last = ksorted[0]
    processed = count[nonzero[0]]  # количество обработанных матриц
    for exp in nonzero[1:]:
        new, last = exp - last, exp  # разница между степенями
        A[idx[processed:]] = torch.matrix_power(A[idx[processed:]], 2 ** new.item())  # возводим
        processed += count[exp]
    return A.reshape(orig_size)  # возвращаем исходную форму батча


# ============================
# Аппроксимация экспоненты Тейлора
# ============================
def taylor_approx(A, deg):
    """
    Вычисляет exp(A) с помощью разложения Тейлора выбранной степени.
    Поддерживаются специальные оптимизации для deg = 1,2,4,8,12,18
    """
    batched = A.ndimension() > 2  # проверяем, батч или одна матрица
    I = torch.eye(A.size(-2), A.size(-1), dtype=A.dtype, device=A.device)  # единичная матрица
    if batched:
        I = I.expand_as(A)  # расширяем для батча

    # Предвычисления степеней матрицы
    if deg >= 2:
        A2 = A @ A
    if deg > 8:
        A3 = A @ A2
    if deg == 18:
        A6 = A3 @ A3

    # ============================
    # Простые степени Тейлора
    # ============================
    if deg == 1:
        return I + A
    elif deg == 2:
        return I + A + 0.5 * A2
    elif deg == 4:
        return I + A + A2 @ (.5 * I + A / 6. + A2 / 24.)
    elif deg == 8:
        # оптимизированный вариант с константами для точности
        SQRT = math.sqrt(177.)
        x3 = 2. / 3.
        a1 = (1. + SQRT) * x3
        x1 = a1 / 88.
        x2 = a1 / 352.
        c0 = (-271. + 29. * SQRT) / (315. * x3)
        c1 = (11. * (-1. + SQRT)) / (1260. * x3)
        c2 = (11. * (-9. + SQRT)) / (5040. * x3)
        c4 = (89. - SQRT) / (5040. * x3 * x3)
        y2 = ((857. - 58. * SQRT)) / 630.
        A4 = A2 @ (x1 * A + x2 * A2)
        A8 = (x3 * A2 + A4) @ (c0 * I + c1 * A + c2 * A2 + c4 * A4)
        return I + A + y2 * A2 + A8
    elif deg == 12:
        # оптимизация для степени 12 с матрицей коэффициентов
        b = torch.tensor(
            [[-1.86023205146205530824e-02, -5.00702322573317714499e-03, -5.73420122960522249400e-01,
              -1.33399693943892061476e-01],
             [4.6, 9.92875103538486847299e-01, -1.32445561052799642976e-01, 1.72990000000000000000e-03],
             [2.11693118299809440730e-01, 1.58224384715726723583e-01, 1.65635169436727403003e-01,
              1.07862779315792429308e-02],
             [0., -1.31810610138301836924e-01, -2.02785554058925905629e-02, -6.75951846863086323186e-03]],
            dtype=A.dtype, device=A.device
        )
        A3 = A @ (A2 if 'A2' in locals() else A)
        q = torch.stack([I, A, A2, A3], dim=-3).unsqueeze_(-4)
        len_batch = A.ndimension() - 2
        q_size = [-1 for _ in range(len_batch)] + [4, -1, -1, -1]
        q = q.expand(*q_size)
        b = b.unsqueeze_(-1).unsqueeze_(-1).expand_as(q)
        q = (b * q).sum(dim=-3)
        if batched:
            qaux = q[..., 2, :, :] + q[..., 3, :, :] @ q[..., 3, :, :]
            return q[..., 0, :, :] + (q[..., 1, :, :] + qaux) @ qaux
        else:
            qaux = q[2] + q[3] @ q[3]
            return q[0] + (q[1] + qaux) @ qaux
    elif deg == 18:
        # оптимизация для степени 18 с матрицей коэффициентов
        b = torch.tensor(
            [[0., -1.00365581030144618291e-01, -8.02924648241156932449e-03, -8.92138498045729985177e-04, 0.],
             [0., 3.97849749499645077844e-01, 1.36783778460411720168e+00, 4.98289622525382669416e-01,
              -6.37898194594723280150e-04],
             [-1.09676396052962061844e+01, 1.68015813878906206114e+00, 5.71779846478865511061e-02,
              -6.98210122488052056106e-03, 3.34975017086070470649e-05],
             [-9.04316832390810593223e-02, -6.76404519071381882256e-02, 6.75961301770459654925e-02,
              2.95552570429315521194e-02, -1.39180257516060693404e-05],
             [0., 0., -9.23364619367118555360e-02, -1.69364939002081722752e-02, -1.40086798182036094347e-05]],
            dtype=A.dtype, device=A.device
        )
        A3 = A @ A2
        A6 = A3 @ A3
        q = torch.stack([I, A, A2, A3, A6], dim=-3).unsqueeze_(-4)
        len_batch = A.ndimension() - 2
        q_size = [-1 for _ in range(len_batch)] + [5, -1, -1, -1]
        q = q.expand(*q_size)
        b = b.unsqueeze_(-1).unsqueeze_(-1).expand_as(q)
        q = (b * q).sum(dim=-3)
        if batched:
            qaux = q[..., 0, :, :] @ q[..., 4, :, :] + q[..., 3, :, :]
            return q[..., 1, :, :] + (q[..., 2, :, :] + qaux) @ qaux
        else:
            qaux = q[0] @ q[4] + q[3]
            return q[1] + (q[2] + qaux) @ qaux


# ============================
# Основная функция expm_taylor
# ============================
def expm_taylor(A, order=14, tol=1e-16):
    """
    Вычисляет exp(A) методом Тейлора с выбранной степенью.

    Параметры:
        A : [n,n] или батч [b,n,n]
        order : int, максимальная степень разложения Тейлора
        tol : порог для "малой нормы"
    """
    if A.ndimension() < 2 or A.size(-2) != A.size(-1):
        raise ValueError('Expected a square matrix or a batch of square matrices')

    squeeze = False
    if A.ndimension() == 2:
        A = A.unsqueeze(0)  # добавляем размер батча
        squeeze = True

    # ============================
    # Тривиальные случаи
    # ============================
    if A.size(-2) == 1:
        X = torch.exp(A)  # просто exp для скаляра
    else:
        normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values  # инф-1 норма

        if (normA == 0.).all():  # нулевая матрица
            I = torch.eye(A.size(-2), A.size(-1), dtype=A.dtype, device=A.device)
            X = I + A
        else:
            # масштабируем матрицу, если норма > максимального theta
            more = normA > thetas_single[-1]
            k = normA.new_zeros(normA.size(), dtype=torch.long)
            k[more] = torch.ceil(torch.log2(normA[more]) - math.log2(thetas_single[-1])).long()

            A_scaled = torch.pow(.5, k.float()).unsqueeze(-1).unsqueeze(-1).expand_as(A) * A

            deg_use = max([d for d in degs if d <= order])  # выбираем подходящую степень

            term_norm = torch.norm(A_scaled)
            I = torch.eye(A.size(-2), A.size(-1), dtype=A.dtype, device=A.device)
            if term_norm < tol:
                X = I + A_scaled  # приближение для маленькой нормы
            else:
                X = taylor_approx(A_scaled, deg_use)

            # возвращаем масштабирование
            X = matrix_power_two_batch(X, k)

    if squeeze:
        return X.squeeze(0)

    return X


# ============================
# Дифференциал exp(A)
# ============================
def differential(A, E, f):
    """
    Вычисляет Df(A) @ E через блочную матрицу:
        | A  E |
        | 0  A |
    """
    n = A.size(-1)
    size_M = list(A.size()[:-2]) + [2 * n, 2 * n]
    M = A.new_zeros(size_M)
    M[..., :n, :n] = A
    M[..., n:, n:] = A
    M[..., :n, n:] = E
    return f(M)[..., :n, n:]  # извлекаем верхний правый блок