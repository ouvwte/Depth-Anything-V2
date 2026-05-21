import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'original_repo'))
from depth_anything_v2.dpt import DepthAnythingV2

def compare_depth_matrices(ref_matrix, test_matrix, test_name):
    """Функция вычитания матриц, расчета метрик и сохранения визуализации"""
    print(f"\n ЗАПУСК: {test_name}")
    
    if ref_matrix.shape != test_matrix.shape:
        print("❌ ТЕСТ ПРОВАЛЕН: Размерности матриц не совпадают!")
        return False

    # Вычитание матриц
    diff_matrix = ref_matrix - test_matrix
    
    # Нормализация для SSIM (0-1)
    ref_norm = np.zeros_like(ref_matrix)
    test_norm = np.zeros_like(test_matrix)
    cv2.normalize(ref_matrix, ref_norm, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(test_matrix, test_norm, 0, 1, cv2.NORM_MINMAX)

    # Метрики
    mse_value = np.mean(diff_matrix ** 2)
    ssim_value = ssim(ref_norm, test_norm, data_range=1.0)
    max_abs_diff = np.max(np.abs(diff_matrix))

    print(f"  [MSE]  Среднеквадратичная ошибка : {mse_value:.8f}")
    print(f"  [SSIM] Структурное сходство      : {ssim_value:.4f} (1.0 = идеал)")

    # Создаем папку для результатов внутри контейнера
    os.makedirs('output', exist_ok=True)
    
    # Подготавливаем матрицу для картинки: приводим разницу к 0-255 (uint8)
    diff_visual = np.zeros_like(diff_matrix, dtype=np.uint8)
    cv2.normalize(diff_matrix, diff_visual, 0, 255, cv2.NORM_MINMAX)
    
    # Применяем цветовую карту (JET) - там где 0 (нет разницы) будет черный, а где есть отклонения - желтый/красный
    diff_colored = cv2.applyColorMap(diff_visual, cv2.COLORMAP_JET)
    
    # В имени файла убираем пробелы и двоеточия
    safe_filename = test_name.replace(" ", "_").replace(":", "")
    save_path = f'output/diff_{safe_filename}.png'
    
    cv2.imwrite(save_path, diff_colored)
    print(f"  [VIS] Карта отклонений сохранена: {save_path}")

    # Пороги
    if mse_value < 1e-4 and ssim_value > 0.95:
        print("\n  ✅ РЕЗУЛЬТАТ: ТЕСТ ПРОЙДЕН. Карты идентичны.")
        return True
    else:
        print("\n  ❌ РЕЗУЛЬТАТ: ТЕСТ ПРОЙДЕН (ОШИБКА ОЖИДАЕМА). Карты различаются.")
        return False

def main():
    print("СИСТЕМА ВАЛИДАЦИИ DOCKER-ОБРАЗА DEPTH-ANYTHING")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nВычислительное устройство: {DEVICE}")

    # Инициализация модели (используется конфигурация Small для простоты и ускорения инференса для тестирования)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vits' 
    
    print("Инициализация модели (загрузка весов)...")
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    print("Модель готова к работе.")

    # Загрузка эталонной матрицы
    ref_depth = np.load('reference_depth.npy')

    # ТЕСТ 1: Положительный (Ожидаем успех)
    img1_path = 'tests/test_image_1.jpg'
    raw_img1 = cv2.imread(img1_path)
    user_depth_1 = model.infer_image(raw_img1)
    
    # Сравниваем картинку 1 с ЭТАЛОНОМ
    test1_passed = compare_depth_matrices(ref_depth, user_depth_1, "Тест 1: Эталонное изображение (Ожидается Успех)")

    # ТЕСТ 2: Негативный (Ожидаем провал)
    img2_path = 'tests/test_image_2.jpg'
    raw_img2 = cv2.imread(img2_path)
    user_depth_2 = model.infer_image(raw_img2)
    
    # СРАВНИВАЕМ КАРТИНКУ 2 С ЭТАЛОНОМ ОТ КАРТИНКИ 1
    # Так как это разные картинки, матрицы глубин будут совершенно разными
    test2_passed = compare_depth_matrices(ref_depth, user_depth_2, "Тест 2: Другое изображение (Ожидается Провал)")

    # ИТОГ
    if test1_passed and not test2_passed:
        print("ИТОГО: ВСЕ ТЕСТЫ ОТРАБОТАЛИ КОРРЕКТНО!")
        print("Модель отличает правильную карту глубины от неправильной.")
        print("Docker-образ полностью рабочий.")
    else:
        print("ИТОГО: ЛОГИКА ТЕСТИРОВАНИЯ НАРУШЕНА!")
        print("(Либо сломалась модель, либо тесты написаны неверно)")

if __name__ == '__main__':
    main()