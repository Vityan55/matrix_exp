# main.py
import time
import math
from expm_adaptive import expm_hybrid as expm_adaptive
from expm_taylor import expm_taylor
from expm_sketch import expm_pade_orthogonal_sketch
from expm_pade import expm_pade

import torch
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
    ("Random 50*50", torch.randn(50, 50)),
    ("Random 75*75", torch.randn(75, 75)),
    ("Random 100*100", torch.randn(100, 100)),
    ("Random 250*250", torch.randn(250, 250)),
    ("Random 500*500", torch.randn(500, 500)),
    ("Random 750*750", torch.randn(750, 750)),
    ("Random 1000*1000", torch.randn(1000, 1000)),
    ("Random 1500*1500", torch.randn(1500, 1500)),
    ("Random 2000*2000", torch.randn(2000, 2000)),
    ("Random 3000*3000", torch.randn(3000, 3000)),
    ("Random 5000*5000", torch.randn(5000, 5000))
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


# Основной тест
def main():
    print(f"Using device: {device}\n")

    for i, (name, A) in enumerate(matrices, 1):
        A = A.to(device, dtype=torch.float64)
        print(f"\n--- Test matrix {i}: {name} ({A.shape}) ---")

        # 1️⃣ Точное значение через PyTorch
        exact, t_exact = measure_time(torch.matrix_exp, A)

        # 2️⃣ Метод Тейлора
        taylor_result, t_taylor = measure_time(expm_taylor, A)
        error_taylor = torch.norm(taylor_result - exact) / torch.norm(exact)

        # 3️⃣ Adaptive ML (старый selector)
        adaptive_result, t_adaptive = measure_time(expm_adaptive, A)
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