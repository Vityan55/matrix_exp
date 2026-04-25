# main.py
import math
import time

import expm_adaptive
import torch
from scipy import sparse

from expm_adaptive import expm_hybrid
from expm_pade import expm_pade
from expm_sketch import expm_pade_orthogonal_sketch
from expm_taylor import expm_taylor

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Тестовые матрицы с названиями
matrices = [
    ("Zero 3x3", torch.zeros(3, 3)),
    ("Diagonal 2x2", torch.tensor([[2., 0.], [0., -1.]])),
    ("Diagonal 3x3", torch.tensor([[1., 0., 0.], [0., 0.5, 0.], [0., 0., -0.2]])),
    ("Rotation 2D pi/4", torch.tensor([[0., -math.pi / 4], [math.pi / 4, 0.]])),
    ("Nilpotent 2x2", torch.tensor([[0., 1.], [0., 0.]])),
    ("Upper Triangular 2x2", torch.tensor([[3., 1.], [0., 2.]])),
    ("Linear system 3x3", torch.tensor([[2., -1., 1.], [0., 3., -1.], [2., 1., 3.]])),
    ("Repeated eigenvalues", torch.tensor([[2., 1.], [0., 2.]])),
    ("3D rotation x-axis", torch.tensor([[0., 0., 0.], [0., 0., -math.pi / 3], [0., math.pi / 3, 0.]])),
    ("Complex eigenvalues", torch.tensor([[0., -1.], [1., 0.]])),
    ("Diagonalizable 3x3", torch.tensor([[1., 2., 0.], [0., 3., 0.], [0., 0., 4.]])),
    ("Jordan block 3x3", torch.tensor([[5., 1., 0.], [0., 5., 1.], [0., 0., 5.]])),
    ("Eigenvalue test 2x2", torch.tensor([[0., 1.], [-1., 0.]])),
    ("2D Lie matrix", torch.tensor([[1., 2.], [2., 1.]])),
    ("Textbook ODE 3x3", torch.tensor([[1., 2., 3.], [3., 2., 1.], [0., 1., 4.]])),
    ("Block diagonal 4x4", torch.block_diag(torch.tensor([[0., -math.pi / 4], [math.pi / 4, 0.]]),
                                               torch.tensor([[2., 0.], [0., -1.]]))),
    ("Fast decay", torch.tensor([[-1., 0., 0.], [0., -2., 0.], [0., 0., -3.]])),
    ("State switching", torch.tensor([[0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]])),
    ("Zero 4x4", torch.zeros(4, 4)),
    ("Upper triangular 4x4", torch.tensor([[1., 2., 3., 4.], [0., 1., 2., 3.], [0., 0., 1., 2.], [0., 0., 0., 1.]])),
    ("Nilpotent 3x3", torch.tensor([[0., 1., 2.], [0., 0., 1.], [0., 0., 0.]])),
    ("Growth/decay 3x3", torch.tensor([[1., 0., 0.], [0., -0.5, 0.], [0., 0., 2.]])),
    ("Jordan 2x2 + diag", torch.tensor([[2., 1., 0.], [0., 2., 0.], [0., 0., 5.]])),
    ("Transition matrix 2x2", torch.tensor([[0., 1.], [-2., -3.]])),
    ("Linear system 4x4", torch.tensor([[1., 2., 0., 0.], [0., 1., 3., 1.], [0., 0., 2., 4.], [1., 0., 1., 2.]])),
    ("Block diagonal mixed", torch.block_diag(torch.tensor([[0., -math.pi / 4], [math.pi / 4, 0.]]),
                                               torch.tensor([[0., 1.], [0., 0.]]),
                                               torch.tensor([[2., 0.], [0., -1.]])))
]

# Добавляем к существующим matrices
large_matrices = [
    ("Diagonal 6x6", torch.diag(torch.tensor([1., 2., 3., -1., -2., -3.], dtype=torch.float64))),
    ("Upper Triangular 6x6", torch.tensor([
        [1., 2., 3., 4., 5., 6.],
        [0., 1., 2., 3., 4., 5.],
        [0., 0., 1., 2., 3., 4.],
        [0., 0., 0., 1., 2., 3.],
        [0., 0., 0., 0., 1., 2.],
        [0., 0., 0., 0., 0., 1.]
    ], dtype=torch.float64)),
    ("Jordan block 6x6", torch.tensor([
        [5., 1., 0., 0., 0., 0.],
        [0., 5., 1., 0., 0., 0.],
        [0., 0., 5., 1., 0., 0.],
        [0., 0., 0., 5., 1., 0.],
        [0., 0., 0., 0., 5., 1.],
        [0., 0., 0., 0., 0., 5.]
    ], dtype=torch.float64)),
    ("Block diagonal 8x8", torch.block_diag(
        torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64),
        torch.tensor([[2., 1.], [0., 3.]], dtype=torch.float64),
        torch.tensor([[0., 1.], [0., 0.]], dtype=torch.float64),
        torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    )),
    ("Linear system 6x6", torch.tensor([
        [1., 2., 0., 0., 1., 0.],
        [0., 1., 3., 0., 0., 1.],
        [0., 0., 2., 4., 0., 0.],
        [1., 0., 1., 2., 1., 0.],
        [0., 1., 0., 0., 3., 1.],
        [0., 0., 1., 1., 0., 2.]
    ], dtype=torch.float64)),
    ("Nilpotent 6x6", torch.tensor([
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0.]
    ], dtype=torch.float64)),
    ("Rotation 3D 8x8 block", torch.block_diag(
        torch.tensor([[0., -math.pi/4], [math.pi/4, 0.]], dtype=torch.float64),
        torch.tensor([[0., -math.pi/6], [math.pi/6, 0.]], dtype=torch.float64),
        torch.tensor([[0., -math.pi/3], [math.pi/3, 0.]], dtype=torch.float64),
        torch.tensor([[0., -math.pi/8], [math.pi/8, 0.]], dtype=torch.float64)
    )),
    ("Damped system 6x6", torch.diag(torch.tensor([-1., -2., -3., -0.5, -1.5, -2.5], dtype=torch.float64))),
    ("State switch 6x6", torch.tensor([
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.],
        [-1., -1., -1., -1., -1., -1.]
    ], dtype=torch.float64)),
    ("Upper Triangular 8x8", torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [0., 1., 2., 3., 4., 5., 6., 7.],
        [0., 0., 1., 2., 3., 4., 5., 6.],
        [0., 0., 0., 1., 2., 3., 4., 5.],
        [0., 0., 0., 0., 1., 2., 3., 4.],
        [0., 0., 0., 0., 0., 1., 2., 3.],
        [0., 0., 0., 0., 0., 0., 1., 2.],
        [0., 0., 0., 0., 0., 0., 0., 1.]
    ], dtype=torch.float64)),

    ("Random 20*20", torch.randn(20, 20)),
    ("Sparse 20*20", torch.tensor(sparse.random(20, 20, density=0.001).toarray(), dtype=torch.float64)),
    ("Random 50*50", torch.randn(50, 50)),
    ("Sparse 50*50", torch.tensor(sparse.random(50, 50, density=0.001).toarray(), dtype=torch.float64)),
    ("Random 75*75", torch.randn(75, 75)),
    ("Random 100*100", torch.randn(100, 100)),
    ("Random 250*250", torch.randn(250, 250)),
    ("Random 500*500", torch.randn(500, 500)),
    ("Sparse 500*500", torch.tensor(sparse.random(500, 500, density=0.001).toarray(), dtype=torch.float64)),
    ("Random 750*750", torch.randn(750, 750)),
    ("Random 1000*1000", torch.randn(1000, 1000)),
    ("Random 1500*1500", torch.randn(1500, 1500)),
    ("Sparse 1500*1500", torch.tensor(sparse.random(1500, 1500, density=0.001).toarray(), dtype=torch.float64)),
    ("Random 2000*2000", torch.randn(2000, 2000)),
    ("Random 3000*3000", torch.randn(3000, 3000)),
    ("Random 5000*5000", torch.randn(5000, 5000)),
    ("Sparse 5000*5000", torch.tensor(sparse.random(5000, 5000, density=0.001).toarray(), dtype=torch.float64))
]

# Добавляем к основному списку
matrices.extend(large_matrices)


def measure_time(func, A):
    """Замер времени выполнения функции"""
    if A.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    result = func(A)
    if A.is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    return result, end - start

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Генерация разных типов матриц
# ============================
def generate_matrix(n, kind):
    """
    Создает матрицу размера n x n определенного типа:
    kind может быть 'random', 'symmetric', 'skew', 'diagonal', ...
    """
    A = torch.randn(n, n, device=DEVICE)  # базовая случайная матрица

    if kind == "random":
        pass  # оставляем как есть
    elif kind == "symmetric":
        A = (A + A.T) / 2  # симметричная матрица
    elif kind == "skew":
        A = (A - A.T) / 2  # кососимметричная матрица
    elif kind == "diagonal":
        A = torch.diag(torch.randn(n, device=DEVICE))  # диагональная матрица
    elif kind == "positive_diag":
        A = torch.diag(torch.abs(torch.randn(n, device=DEVICE)) + 1e-2)  # положительная диагональная
    elif kind == "nilpotent":
        A = torch.triu(A, 1)  # верхняя треугольная с нулями на диагонали
    elif kind == "ill_conditioned":
        # сильно плохо обусловленная матрица
        U, _, V = torch.linalg.svd(A)
        s = torch.logspace(0, -8, n, device=DEVICE)
        A = U @ torch.diag(s) @ V
    elif kind == "hilbert":
        i = torch.arange(n, device=DEVICE).float()
        j = i.view(-1, 1)
        A = 1.0 / (i + j + 1)  # матрица Гильберта
    elif kind == "permutation":
        idx = torch.randperm(n, device=DEVICE)
        A = torch.eye(n, device=DEVICE)[idx]  # случайная перестановка
    elif kind == "toeplitz":
        c = torch.randn(n, device=DEVICE)
        r = torch.randn(n, device=DEVICE)
        A = torch.zeros(n, n, device=DEVICE)
        for i in range(n):
            for j in range(n):
                A[i, j] = c[i - j] if i >= j else r[j - i]
    elif kind == "circulant":
        v = torch.randn(n, device=DEVICE)
        A = torch.stack([torch.roll(v, shifts=i) for i in range(n)])
    elif kind == "upper_triangular":
        A = torch.triu(A)  # верхняя треугольная
    elif kind == "lower_triangular":
        A = torch.tril(A)  # нижняя треугольная
    elif kind == "spd":  # симметричная положительно определенная
        A = torch.randn(n, n, device=DEVICE)
        A = A @ A.T + n * torch.eye(n, device=DEVICE)
    elif kind == "jordan":
        A = torch.diag(torch.ones(n, device=DEVICE)) + torch.diag(torch.ones(n - 1, device=DEVICE), 1)
    elif kind == "block_diag":
        blocks = [torch.randn(n // 2, n // 2, device=DEVICE), torch.randn(n - n // 2, n - n // 2, device=DEVICE)]
        A = torch.block_diag(*blocks)
    return A


def build_dataset(samples_per_kind=2):
    """
    Создает X и y для обучения модели
    X - признаки матриц
    y - метка лучшего метода
    """
    sizes = [4, 16, 64, 128, 256, 512, 1024]  # размеры матриц
    kinds = [
        "random", "symmetric", "skew", "diagonal", "positive_diag",
        "nilpotent", "ill_conditioned", "hilbert", "permutation",
        "toeplitz", "circulant", "upper_triangular", "lower_triangular",
        "spd", "jordan", "block_diag"
    ]

    dataset = []

    for n in sizes:
        for kind in kinds:
            scales = [0.1, 1.0, 10.0, 100.0]  # разные масштабы значений
            sparsity = [0.0, 0.5, 0.9]  # плотность/разреженность матриц

            for scale in scales:
                for spar in sparsity:
                    if n < 500:
                        samples = samples_per_kind  # маленькие матрицы - больше образцов
                    else:
                        samples = 1  # большие матрицы - 1 образец

                    for _ in range(samples):
                        A = generate_matrix(n, kind) * scale  # масштабируем матрицу

                        if spar > 0:
                            # делаем матрицу разреженной
                            mask = torch.rand(n, n, device=DEVICE) > spar
                            A = A * mask.float()

                        dataset.append((f"{kind}_{n}", A))
    return dataset

def benchmark_full(dataset):
    results = []

    for idx, (name, A) in enumerate(dataset):
        A = A.to(DEVICE, dtype=torch.float64)

        kind = name.split("_")[0]
        size = A.shape[0]

        print(f"[{idx+1}/{len(dataset)}] {name} {A.shape}")

        # --- ground truth ---
        exact, _ = measure_time(torch.matrix_exp, A)

        def safe_rel_err(X, exact):
            if not torch.isfinite(X).all():
                return np.nan
            err = torch.norm(X - exact) / torch.norm(exact)
            return err.item() if torch.isfinite(err) else np.nan

        # --- Taylor ---
        taylor, t_taylor = measure_time(expm_taylor, A)
        e_taylor = safe_rel_err(taylor, exact)

        # --- Pade ---
        pade, t_pade = measure_time(expm_pade, A)
        e_pade = safe_rel_err(pade, exact)

        # --- Sketch ---
        sketch, t_sketch = measure_time(
            lambda X: expm_pade_orthogonal_sketch(X, (5, min(64, size // 2))),
            A
        )
        e_sketch = safe_rel_err(sketch, exact)

        # --- Adaptive ---
        adaptive, t_adaptive = measure_time(expm_hybrid, A)
        e_adaptive = safe_rel_err(adaptive, exact)

        results.append({
            "type": kind,
            "size": size,

            "taylor_time": t_taylor,
            "taylor_err": e_taylor,

            "pade_time": t_pade,
            "pade_err": e_pade,

            "sketch_time": t_sketch,
            "sketch_err": e_sketch,

            "adaptive_time": t_adaptive,
            "adaptive_err": e_adaptive,

            "adaptive_choice": expm_adaptive.last_method
        })

        print("errors:",
          "adaptive =", e_adaptive,
          "sketch =", e_sketch)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    return df

# =========================
# Русские названия
# =========================
METHOD_NAMES = {
    "taylor": "Тейлор",
    "pade": "Паде",
    "sketch": "Скетч",
    "adaptive": "Адаптивный"
}

TYPE_NAMES = {
    "random": "Случайная",
    "symmetric": "Симметричная",
    "skew": "Кососимметричная",
    "diagonal": "Диагональная",
    "positive_diag": "Положит. диагональная",
    "nilpotent": "Нильпотентная",
    "ill_conditioned": "Плохо обусловленная",
    "hilbert": "Гильберта",
    "permutation": "Перестановочная",
    "toeplitz": "Тёплицева",
    "circulant": "Циркулянтная",
    "upper_triangular": "Верхнетреугольная",
    "lower_triangular": "Нижнетреугольная",
    "spd": "Положительно определённая",
    "jordan": "Жорданова",
    "block_diag": "Блочно-диагональная"
}

# =========================
# Цвета (единые для всех графиков)
# =========================
METHOD_COLORS = {
    "taylor": "blue",
    "pade": "green",
    "sketch": "orange",
    "adaptive": "red"
}

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Рисунок 1 — Время выполнения vs размер матрицы
def plot_time_vs_size(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    grouped = df.groupby("size").mean(numeric_only=True)

    plt.figure(figsize=(10,6))

    for method in ["taylor", "pade", "sketch", "adaptive"]:
        y = grouped[f"{method}_time"]
        mask = (~y.isna())
        plt.plot(
            grouped.index[mask],
            y[mask],
            label=METHOD_NAMES[method],
            color=METHOD_COLORS[method]
        )

    plt.xlabel("Размер матрицы")
    plt.ylabel("Время (сек)")
    plt.title("Зависимость времени выполнения от размера матрицы (усреднение по типам)")
    plt.yscale("log")
    plt.grid()
    plt.legend(title="Метод")

    plt.figtext(0.5, -0.05, "Рисунок 1 — Зависимость времени выполнения от размера матрицы",
                ha="center", fontsize=10)

    plt.show()


# Рисунок 2 — Ошибка vs размер матрицы
def plot_error_vs_size(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    grouped = df.groupby("size").median(numeric_only=True)

    plt.figure(figsize=(10,6))

    for method in ["taylor", "pade", "sketch", "adaptive"]:
        y = grouped[f"{method}_err"]
        mask = (y > 0) & (~y.isna())
        plt.plot(
            grouped.index[mask],
            y[mask],
            label=METHOD_NAMES[method],
            color=METHOD_COLORS[method]
        )

    plt.xlabel("Размер матрицы")
    plt.ylabel("Относительная ошибка")
    plt.title("Зависимость ошибки от размера матрицы")
    plt.yscale("log")
    plt.grid()
    plt.legend(title="Метод")

    plt.figtext(0.5, -0.05, "Рисунок 2 — Зависимость относительной ошибки от размера матрицы",
                ha="center", fontsize=10)

    plt.show()


# Рисунок 3 — Выбор адаптивного метода по типам матриц
def plot_adaptive_by_type(df):
    df["type_ru"] = df["type"].map(TYPE_NAMES)

    pivot = pd.crosstab(df["type_ru"], df["adaptive_choice"], normalize="index")

    pivot.rename(columns=METHOD_NAMES).plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=[METHOD_COLORS[k] for k in pivot.columns]
    )

    pivot.plot(kind="bar", stacked=True, figsize=(12, 6))

    plt.title("Распределение выбора адаптивного метода по типам матриц")
    plt.ylabel("Доля")
    plt.xlabel("Тип матрицы")
    plt.legend(title="Метод")
    plt.grid()

    plt.figtext(0.5, -0.08,
                "Рисунок 3 — Доли выбора методов адаптивным алгоритмом для различных типов матриц",
                ha="center", fontsize=10)

    plt.show()


# Рисунок 4 — Выбор адаптивного метода по размерам
def plot_adaptive_by_size(df):
    pivot = pd.crosstab(df["size"], df["adaptive_choice"], normalize="index")

    pivot.rename(columns=METHOD_NAMES).plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=[METHOD_COLORS[k] for k in pivot.columns]
    )

    pivot.plot(kind="bar", stacked=True, figsize=(12, 6))

    plt.title("Распределение выбора адаптивного метода по размерам матриц")
    plt.ylabel("Доля")
    plt.xlabel("Размер матрицы")
    plt.legend(title="Метод")
    plt.grid()

    plt.show()


# Рисунок 5 — Общая статистика выбора методов
def plot_adaptive_global(df):
    counts = df["adaptive_choice"].value_counts(normalize=True)

    counts.rename(index=METHOD_NAMES).plot(
        kind="bar",
        figsize=(6, 4),
        color=[METHOD_COLORS[k] for k in counts.index]
    )

    counts.plot(kind="bar", figsize=(6, 4))

    plt.title("Общее распределение использования методов")
    plt.ylabel("Доля")
    plt.xlabel("Метод")
    plt.grid()

    plt.figtext(0.5, -0.15,
                "Рисунок 5 — Общая доля использования методов адаптивным алгоритмом",
                ha="center", fontsize=10)

    plt.show()


# Рисунок 6 — Качество выбора адаптивного метода
def plot_choice_quality(df):
    plt.figure(figsize=(8,6))

    for method in ["taylor", "pade", "sketch"]:
        subset = df[df["adaptive_choice"] == method]

        plt.scatter(
            subset[f"{method}_time"],
            subset[f"{method}_err"],
            label=METHOD_NAMES[method],
            alpha=0.6,
            color=METHOD_COLORS[method]
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Время (сек)")
    plt.ylabel("Относительная ошибка")
    plt.title("Качество решений адаптивного алгоритма (время vs ошибка)")
    plt.legend(title="Выбранный метод")
    plt.grid()

    plt.figtext(0.5, -0.05,
                "Рисунок 6 — Соотношение времени и ошибки для выбранных адаптивным алгоритмом методов",
                ha="center", fontsize=10)

    plt.show()
# Основной тест
def main():
    print(f"Using device: {device}\n")

    print("\n===== FULL BENCHMARK =====")

    dataset = build_dataset(samples_per_kind=2)
    df = benchmark_full(dataset)

    plot_time_vs_size(df)
    plot_error_vs_size(df)
    plot_adaptive_global(df)
    plot_adaptive_by_size(df)
    plot_adaptive_by_type(df)
    plot_choice_quality(df)

    for i, (name, A) in enumerate(matrices, 1):
        A = A.to(device, dtype=torch.float64)
        print(f"\n--- Test matrix {i}: {name} ({A.shape}) ---")

        # 1️⃣ Точное значение через PyTorch
        exact, t_exact = measure_time(torch.matrix_exp, A)

        # 2️⃣ Метод Тейлора
        taylor_result, t_taylor = measure_time(expm_taylor, A)
        error_taylor = torch.norm(taylor_result - exact) / torch.norm(exact)

        # 3️⃣ Adaptive ML (старый selector)
        adaptive_result, t_adaptive = measure_time(expm_hybrid, A)
        error_adaptive = torch.norm(adaptive_result - exact) / torch.norm(exact)

        # 4️⃣ Orthogonal Sketch + Pade
        sketch_params = (5, min(64, max(8, A.shape[0] // 2)))
        sketch_result, t_sketch = measure_time(
            lambda X: expm_pade_orthogonal_sketch(X, sketch_params),
            A
        )
        error_sketch = torch.norm(sketch_result - exact) / torch.norm(exact)

        X_pade, t_pade = measure_time(expm_pade, A)
        err_pade = torch.norm(X_pade - exact) / torch.norm(exact)

        # Вывод
        print(f"Time torch.matrix_exp : {t_exact:.6f} sec")
        print(f"Time Taylor           : {t_taylor:.6f} sec, rel_error = {error_taylor:.3e}")
        print(f"Time Pade             : {t_pade:.6f} sec, rel_error = {err_pade:.3e}")
        print(f"Time Sketch           : {t_sketch:.6f} sec, rel_error = {error_sketch:.3e}")
        print(f"Time Adaptive         : {t_adaptive:.6f} sec, rel_error = {error_adaptive:.3e}")


if __name__ == "__main__":
    main()