# expm_adaptive.py
import torch  # для работы с тензорами и GPU

from expm_pade import expm_pade  # метод экспоненты через рациональную аппроксимацию Паде
from expm_sketch import expm_pade_orthogonal_sketch
from expm_taylor import expm_taylor  # метод экспоненты через разложение Тейлора
from features import extract_features  # функция извлечения признаков матрицы
from train_selector import SelectorHybridModel  # наша обученная модель выбора метода

# -----------------------------
# Определяем устройство вычислений
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # если есть GPU — используем его

# -----------------------------
# Загружаем обученную модель выбора метода
# -----------------------------`
checkpoint = torch.load("selector_hybrid_new.pt", map_location=DEVICE)  # загружаем веса и статистику
model = SelectorHybridModel().to(DEVICE)  # создаем объект модели и отправляем на устройство
model.load_state_dict(checkpoint["model"])  # загружаем обученные веса
model.eval()  # режим inference (отключаем градиенты, dropout и др.)
mean = checkpoint["mean"].to(DEVICE)  # среднее для нормализации признаков
std = checkpoint["std"].to(DEVICE)  # стандартное отклонение для нормализации


# -----------------------------
# Функция гибридного вычисления exp(A)
# -----------------------------
@torch.no_grad()  # отключаем вычисление градиентов, чтобы экономить память и ускорить inference
def expm_hybrid(A):
    """
    Вычисляет exp(A) с помощью гибридного метода:
    - извлекает признаки матрицы,
    - нормализует их,
    - предсказывает оптимальный метод с обученной моделью,
    - применяет выбранный метод (Тейлор или Паде) с предсказанным порядком.

    Параметры:
        A: torch.Tensor, квадратная матрица [n,n]
    Возвращает:
        X: torch.Tensor, exp(A)
    """

    # -----------------------------
    # Извлекаем признаки матрицы
    # -----------------------------
    feats = extract_features(A)  # базовые признаки: нормы, след, симметрия, разреженность, спектральный радиус

    # -----------------------------
    # Нормализация признаков (среднее и std из обучения)
    # -----------------------------
    X = (feats - mean) / std

    # -----------------------------
    # Предсказание метода
    # -----------------------------
    logits = model(X.unsqueeze(0))  # добавляем batch-измерение
    probs = torch.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)

    #if conf.item() < 0.55:
    #    return expm64(A.double())

    # -----------------------------
    # Соответствие индекса модели и метода + порядок
    # -----------------------------
    mapping = {
        0: ("Taylor", 4),
        1: ("Taylor", 8),
        2: ("Taylor", 12),
        3: ("Pade", 3),
        4: ("Pade", 5),
        5: ("Pade", 7),
        6: ("Pade", 9),
        7: ("Pade", 13),
        8: ("Sketch", (5, 32)),
        9: ("Sketch", (7, 64)),
        10: ("Sketch", (9, 96)),
    }

    name, params = mapping[pred.item()]
    print(f"[Adaptive] method={name}, params={params}, conf={conf.item():.3f}")

    if name == "Taylor":
        return expm_taylor(A, params)
    elif name == "Pade":
        return expm_pade(A, params)
    else:
        return expm_pade_orthogonal_sketch(A, params)