# train_selector_hybrid.py

import time  # для измерения времени выполнения функций
import torch  # PyTorch для работы с матрицами и нейросетью
import torch.nn as nn  # для построения нейронной сети
import torch.optim as optim  # оптимизаторы (Adam и др.)
from torch.utils.data import DataLoader, TensorDataset  # удобный DataLoader для батчей

from features import extract_features  # функция извлечения признаков из матриц
from expm_taylor import expm_taylor  # твой метод Тейлора для exp(A)
from expm_pade import expm_pade  # твой метод Паде для exp(A)

# ============================
# Устройство для вычислений: GPU или CPU
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# Модель нейросети
# ============================
class SelectorHybridModel(nn.Module):
    """
    Нейросеть, которая на основе признаков матрицы предсказывает
    какой метод (Taylor/Pade) и параметры лучше использовать
    """
    def __init__(self, input_dim=12, output_dim=8):
        super().__init__()
        # Сеточная структура:
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),  # нормализация входных признаков
            nn.Linear(input_dim, 128),  # первый линейный слой 12 -> 128
            nn.ReLU(),  # активация ReLU
            nn.Dropout(0.05),  # регуляризация (случайное обнуление 5% нейронов)
            nn.Linear(128, 64),  # второй линейный слой 128 -> 64
            nn.ReLU(),  # активация ReLU
            nn.Linear(64, output_dim)  # выходной слой 64 -> 8 классов (методов)
        )

    def forward(self, x):
        return self.net(x)  # прямой проход


# ============================
# Функция измерения времени
# ============================
def measure_time(func, *args):
    """
    Измеряет время выполнения функции func(*args)
    Синхронизирует CUDA, если используется GPU
    """
    if DEVICE == "cuda":
        torch.cuda.synchronize()  # ждать, пока GPU завершит работу
    t0 = time.perf_counter()  # старт таймера
    result = func(*args)  # вызов функции
    if DEVICE == "cuda":
        torch.cuda.synchronize()  # синхронизация GPU
    t1 = time.perf_counter()  # конец таймера
    return result, t1 - t0  # возвращаем результат и время


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


# ============================
# Выбор наилучшего метода
# ============================
def evaluate_methods(A, tol=1e-6):
    """
    Сравнивает методы Тейлора и Паде с истинной exp(A)
    Возвращает индекс "лучшего метода" в нашей схеме
    """
    true = torch.linalg.matrix_exp(A.double())  # "истинная" экспонента
    norm_true = max(torch.norm(true), 1e-12)
    candidates = []

    # Проверка Тейлора
    for order in [6, 10, 14]:
        T, t = measure_time(expm_taylor, A, order)
        if torch.norm(T - true)/norm_true < tol:
            candidates.append((t, order))

    # Проверка Паде
    for order in [3,5,7,9,13]:
        P, t = measure_time(expm_pade, A, order)
        if torch.norm(P - true)/norm_true < tol:
            candidates.append((t, order+100))  # +100 чтобы отличать от Тейлора

    if not candidates:
        return 7  # fallback
    candidates.sort()  # сортировка по времени
    best = candidates[0][1]

    # отображение на 0..7 классы
    mapping = {6:0, 10:1, 14:2, 103:3,105:4,107:5,109:6,113:7}
    return mapping[best]


# ============================
# Генерация датасета для обучения
# ============================
def build_dataset(samples_per_kind=5):
    """
    Создает X и y для обучения модели
    X - признаки матриц
    y - метка лучшего метода
    """
    sizes = [2,4,8,16,32,64,128,256,512,1024,2048]  # размеры матриц
    kinds = [
        "random", "symmetric", "skew", "diagonal", "positive_diag",
        "nilpotent", "ill_conditioned", "hilbert", "permutation",
        "toeplitz", "circulant", "upper_triangular", "lower_triangular",
        "spd", "jordan", "block_diag"
    ]
    X, y = [], []

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

                        feats = extract_features(A)  # извлекаем признаки
                        # добавляем дополнительные признаки
                        A2 = A @ A
                        A3 = A2 @ A
                        feats_ext = torch.cat([feats, torch.tensor([
                            torch.trace(A2).float(),  # след квадрата
                            torch.trace(A3).float(),  # след куба
                            torch.linalg.norm(A2, 'fro').float()  # норма Фробениуса A²
                        ], device=A.device)])

                        label = evaluate_methods(A)  # находим лучший метод
                        X.append(feats_ext)
                        y.append(label)

    X = torch.stack(X)  # превращаем список в тензор
    y = torch.tensor(y)
    return X, y


# ============================
# Нормализация признаков
# ============================
def normalize(X):
    """
    Нормализует X (по признакам) и возвращает mean/std

    X : torch.Tensor
        Тензор размерности [num_samples, num_features], где каждая строка
        — это объект (матрица), а каждый столбец — признак.

    Возвращает:
        X : torch.Tensor
            Нормализованный тензор с нулевым средним и единичной дисперсией по каждому признаку
        mean : torch.Tensor
            Средние значения признаков (для обратного преобразования или сохранения)
        std : torch.Tensor
            Стандартные отклонения признаков (для обратного преобразования или сохранения)
    """

    mean = X.mean(dim=0)
    # X.mean(dim=0) вычисляет среднее по строкам, оставляя один столбец на признак

    std = X.std(dim=0) + 1e-8
    # X.std(dim=0) вычисляет стандартное отклонение (корень из дисперсии)

    X = (X - mean) / std
    # Для каждого признака вычитаем среднее и делим на std:
    # x_i_normalized = (x_i - mean_i) / std_i
    # После этого каждый признак:
    #   - имеет среднее примерно 0
    #   - стандартное отклонение 1
    # Это помогает нейросети быстрее и стабильнее обучаться:
    #   - градиенты не “скачут” из-за разных масштабов признаков
    #   - обучение становится более устойчивым

    return X, mean, std

# ============================
# Обучение модели
# ============================
def train(batch_size=64, epochs=100):
    """
    Основной цикл обучения модели
    """
    X, y = build_dataset(samples_per_kind=10)  # генерируем датасет
    X, mean, std = normalize(X)  # нормализация
    dataset = TensorDataset(X, y)  # создаем Dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # DataLoader

    model = SelectorHybridModel().to(DEVICE)  # модель на GPU/CPU
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # оптимизатор Adam
    loss_fn = nn.CrossEntropyLoss()  # функция потерь для классификации

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:  # батчи
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()  # обнуляем градиенты
            out = model(xb)  # прямой проход
            loss = loss_fn(out, yb)  # вычисляем loss
            loss.backward()  # обратное распространение
            optimizer.step()  # шаг оптимизатора
            total_loss += loss.item() * xb.size(0)  # суммируем loss

        # выводим средний loss по эпохе
        print(f"epoch {epoch} loss {total_loss/len(dataset):.6f}")

    # сохраняем модель, mean и std
    torch.save({
        "model": model.state_dict(),
        "mean": mean,
        "std": std
    }, "selector_hybrid.pt")


# ============================
# Точка входа
# ============================
if __name__ == "__main__":
    train()