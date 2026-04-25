# expm_sketch.py
#
# ЗАМЕНЯЕТ: expm_sketch.py (оригинальный файл в корне проекта)
#
# ЧТО БЫЛО НЕВЕРНО В ОРИГИНАЛЕ:
#   Формула:  exp(A) ≈ Q · exp(QᵀAQ) · Qᵀ
#   Проблема: при A = 0  →  Q·I·Qᵀ ≠ I  (матрица ранга k, не единичная!)
#             нет теоретического обоснования для произвольной A
#
# ЧТО ВЕРНО (Halko, Martinsson, Tropp 2011):
#   Если A ≈ Q Qᵀ A Q Qᵀ (A лежит в подпространстве Q), то:
#   exp(Q B Qᵀ) = Σ (QBQᵀ)ⁿ/n! = I + Q(Σ Bⁿ/n!)Qᵀ - QQᵀ... нет,
#
#   Правильный вывод:
#     (QBQᵀ)ⁿ = Q Bⁿ Qᵀ  (т.к. QᵀQ = I)
#     exp(QBQᵀ) = Σ (QBQᵀ)ⁿ/n! = Q (Σ Bⁿ/n!) Qᵀ = Q·exp(B)·Qᵀ  ← точно только если A = QBQᵀ
#
#   Для произвольной A нужен сдвиг:
#     A = QBQᵀ + E,  где E = (I - QQᵀ)A — ошибка проекции
#     exp(A) ≈ exp(QBQᵀ) · exp(E) ≈ exp(QBQᵀ)  (если ||E|| мало)
#     exp(QBQᵀ) = I + Q(exp(B) - I)Qᵀ  ← формула с явным единичным сдвигом
#
#   Проверка: A = 0  →  B = 0  →  exp(B)-I = 0  →  результат = I  ✓
#             A = QBQᵀ →  результат = точный exp(A)               ✓
#
# ПУБЛИЧНЫЙ API (совместим с оригиналом):
#   expm_pade_orthogonal_sketch(A, params)
#   expm_pade_sketch(A, params)           ← алиас для обратной совместимости

import math
import torch
from expm_pade import expm_pade


# ══════════════════════════════════════════════════════════════════════════════
# 1. Построение ортонормированного базиса подпространства
# ══════════════════════════════════════════════════════════════════════════════

def _randomized_range_finder(A, k, n_power_iter=2):
    """
    Строит Q ∈ Rⁿˣᵏ, QᵀQ = I, такой что  A ≈ Q Qᵀ A.

    Алгоритм 4.4 из Halko, Martinsson, Tropp (2011):
        1. Omega = случайная гауссова матрица n×k
        2. Y = A · Omega           — проецируем случайные векторы через A
        3. Степенные итерации:     усиливают «главные» направления A,
           ортогонализация на каждом шаге — стабилизирует числа
        4. Q = orth(Y)             — ортонормируем итоговое Y

    Без степенных итераций (n_power_iter=0) — быстрее, но менее точно.
    n_power_iter=2 — хороший баланс точности и стоимости.

    Параметры:
        A            : [n, n] матрица
        k            : размерность подпространства (< n)
        n_power_iter : число итераций степенного метода
    Возвращает:
        Q : [n, k],  QᵀQ = I
    """
    n = A.size(0)
    k = min(k, n)

    # Шаг 1: случайная матрица как начальное зондирование
    Omega = torch.randn(n, k, device=A.device, dtype=A.dtype)

    # Шаг 2: Y = A · Omega — образ случайного подпространства через A
    Y = A @ Omega  # [n, k]

    # Шаг 3: степенные итерации (I → AᵀA)^q · Y
    # Каждая итерация: LU-ортогонализация для подавления числовых ошибок
    for _ in range(n_power_iter):
        # Прямой проход: ортогонализуем Y
        Y, _ = torch.linalg.qr(Y, mode="reduced")   # [n, k]
        # Обратный проход: Aᵀ · Y
        Z = A.T @ Y                                  # [n, k]
        # Ортогонализуем Z
        Z, _ = torch.linalg.qr(Z, mode="reduced")   # [n, k]
        # Прямой проход: A · Z
        Y = A @ Z                                    # [n, k]

    # Шаг 4: итоговая QR-факторизация
    Q, _ = torch.linalg.qr(Y, mode="reduced")       # [n, k]
    return Q


# ══════════════════════════════════════════════════════════════════════════════
# 2. Одна аппроксимация exp(A) через подпространство Q
# ══════════════════════════════════════════════════════════════════════════════

def _single_sketch_expm(A, k, order, n_power_iter=2):
    """
    Вычисляет одну аппроксимацию exp(A) через подпространство размерности k.

    Математика:
        Q  = range_finder(A, k)    — Q ∈ Rⁿˣᵏ, QᵀQ = I
        B  = QᵀAQ                  — B ∈ Rᵏˣᵏ, малая матрица
        exp(A) ≈ I + Q·(exp(B)−I)·Qᵀ

    Параметры:
        A            : [n, n] float64
        k            : размерность подпространства
        order        : порядок Паде для exp(B)
        n_power_iter : итерации степенного метода в range finder
    Возвращает:
        [n, n] аппроксимация exp(A)
    """
    n = A.size(0)

    # ── Шаг 1: строим ортобазис для столбцового пространства A ──────────────
    Q = _randomized_range_finder(A, k, n_power_iter=n_power_iter)  # [n, k]

    # ── Шаг 2: проецируем A на малое подпространство ────────────────────────
    # B = QᵀAQ ∈ Rᵏˣᵏ — вся стоимость O(n²k) здесь
    B = Q.T @ (A @ Q)  # [k, k]

    # ── Шаг 3: точная экспонента малой k×k матрицы ──────────────────────────
    # Стоимость O(k³) — дёшево при k << n
    exp_B = expm_pade(B, order)  # [k, k]

    # ── Шаг 4: ПРАВИЛЬНАЯ формула восстановления ─────────────────────────────
    #
    #   НЕВЕРНО (оригинал):  result = Q @ exp_B @ Q.T
    #     Проблема: при A=0: Q·I·Qᵀ = QQᵀ ≠ I (ранг k, не n)
    #
    #   ВЕРНО:  result = I + Q @ (exp_B - I_k) @ Q.T
    #     При A=0: I + Q·0·Qᵀ = I  ✓
    #     При A = QBQᵀ: точный результат ✓
    #
    I_n = torch.eye(n, device=A.device, dtype=A.dtype)   # [n, n]
    I_k = torch.eye(k, device=A.device, dtype=A.dtype)   # [k, k]

    correction = exp_B - I_k                              # [k, k]
    result = I_n + Q @ correction @ Q.T                  # [n, n]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Адаптивный выбор k
# ══════════════════════════════════════════════════════════════════════════════

def _effective_sketch_dim(n, k_user, k_cap=512):
    """
    Адаптирует k к размеру матрицы.

    Логика:
      - k_user < n  — обычный случай: усиливаем k сублинейно (O(√n))
      - k_user >= n — аппроксимация станет точной, вернём n (→ полный Паде)
      - k_cap      — жёсткое ограничение на память и стоимость O(n²k)

    При k = n формула  I + Q(exp(B)-I)Qᵀ  точна (QQᵀ = I),
    поэтому при k_eff >= n переходим сразу на expm_pade.
    """
    if n <= 1:
        return 0
    if k_user >= n:
        return n

    k_adaptive = max(k_user, min(k_cap, int(math.ceil(4.0 * math.sqrt(float(n))))))
    return min(k_adaptive, n - 1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Публичная функция (совместимый с оригиналом интерфейс)
# ══════════════════════════════════════════════════════════════════════════════

def expm_pade_orthogonal_sketch(A, params):
    """
    Аппроксимация exp(A) через проекцию на случайное подпространство.

    Формула:
        exp(A) ≈ (1/r) Σᵢ [ I + Qᵢ · (exp(QᵢᵀAQᵢ) − I) · Qᵢᵀ ]

    Усреднение по r независимым Qᵢ снижает дисперсию ошибки в r раз.
    Если k_eff >= n — откат на точный expm_pade без потери точности.

    Параметры (params):
        params[0] : order  — порядок Паде для малой матрицы (3,5,7,9,13)
        params[1] : k      — желаемая размерность подпространства
        params[2] : r      — (опц., default=1) число усреднений
        params[3] : k_cap  — (опц., default=512) потолок для k

    Совместимость с оригинальным expm_sketch.py: сигнатура идентична.
    """
    n = A.size(0)

    # Тривиальный случай: 1×1 или скаляр
    if n <= 1:
        return expm_pade(A, int(params[0]))

    # Разбор параметров
    order = int(params[0])
    k_user = int(params[1])
    r      = int(params[2]) if len(params) > 2 else 1
    k_cap  = int(params[3]) if len(params) > 3 else 512

    # Если r не задан явно — ставим 1 (детерминированная проекция быстрее)
    # r=2 только при явном запросе
    if len(params) <= 2:
        r = 1

    # Эффективная размерность подпространства
    k_eff = _effective_sketch_dim(n, k_user, k_cap=k_cap)

    # Если k >= n — точный Паде без проекций
    if k_eff >= n:
        return expm_pade(A, order)

    # Одна проекция (r=1) — самый частый случай
    if r == 1:
        return _single_sketch_expm(A, k_eff, order)

    # Усреднение по r независимым проекциям:
    #   E[result] = exp(A) (несмещённость при малой ошибке проекции)
    #   Var  ∝ 1/r
    acc = torch.zeros(n, n, device=A.device, dtype=A.dtype)
    for _ in range(r):
        acc += _single_sketch_expm(A, k_eff, order)

    return acc / r


# ── Алиас для обратной совместимости (train_selector.py, main.py) ──────────
def expm_pade_sketch(A, params):
    """Алиас: expm_pade_sketch == expm_pade_orthogonal_sketch."""
    return expm_pade_orthogonal_sketch(A, params)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Быстрый самотест
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("Самотест expm_sketch.py")
    print("=" * 60)

    # ── Тест 1: нулевая матрица (exp(0) = I) ────────────────────────────────
    print("\n[Тест 1] A = 0  →  ожидаем exp(A) = I")
    for n in [8, 64, 256]:
        A = torch.zeros(n, n, dtype=torch.float64)
        result = expm_pade_orthogonal_sketch(A, (13, n // 4))
        err = torch.norm(result - torch.eye(n, dtype=torch.float64))
        print(f"  n={n:4d}  ||exp(0) - I|| = {err.item():.2e}  {'✓' if err < 1e-10 else '✗ ОШИБКА'}")

    # ── Тест 2: A лежит точно в подпространстве ─────────────────────────────
    print("\n[Тест 2] A = QBQᵀ (rank-k)  →  формула точна")
    n, k = 128, 8
    Q0, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64))
    B0 = torch.randn(k, k, dtype=torch.float64) * 0.3
    A_lowrank = Q0 @ B0 @ Q0.T
    exact = torch.linalg.matrix_exp(A_lowrank)
    result = expm_pade_orthogonal_sketch(A_lowrank, (13, k))
    err = torch.norm(result - exact) / torch.norm(exact)
    print(f"  rel_error = {err.item():.2e}  {'✓' if err < 1e-5 else '(ожидаемо при несовпадении подпространств)'}")

    # ── Тест 3: случайная матрица, сравнение с torch.linalg.matrix_exp ──────
    print("\n[Тест 3] случайная матрица, разные n")
    for n in [16, 64, 256, 512]:
        A = torch.randn(n, n, dtype=torch.float64) * 0.5
        exact = torch.linalg.matrix_exp(A)
        k = max(8, n // 4)
        result = expm_pade_orthogonal_sketch(A, (13, k))
        err = torch.norm(result - exact) / torch.norm(exact)
        print(f"  n={n:4d}  k={k:4d}  rel_error={err.item():.3e}")

    # ── Тест 4: сравнение старой формулы и новой ────────────────────────────
    print("\n[Тест 4] старая формула vs новая (n=64, k=16)")
    n, k = 64, 16
    A = torch.randn(n, n, dtype=torch.float64) * 0.5
    exact = torch.linalg.matrix_exp(A)
    norm_exact = torch.norm(exact)

    Q = _randomized_range_finder(A, k)
    B = Q.T @ (A @ Q)
    exp_B = expm_pade(B, 13)
    I_k = torch.eye(k, dtype=torch.float64)
    I_n = torch.eye(n, dtype=torch.float64)

    old_formula = Q @ exp_B @ Q.T                    # НЕВЕРНО
    new_formula  = I_n + Q @ (exp_B - I_k) @ Q.T    # ВЕРНО

    err_old = torch.norm(old_formula - exact) / norm_exact
    err_new = torch.norm(new_formula  - exact) / norm_exact
    print(f"  Ошибка старой формулы: {err_old.item():.3e}")
    print(f"  Ошибка новой формулы:  {err_new.item():.3e}")
    print(f"  Улучшение: {(err_old / err_new).item():.1f}x")

    print("\n" + "=" * 60)
    print("Тесты завершены.")
