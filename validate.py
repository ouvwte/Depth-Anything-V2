import cv2
import torch
import numpy as np
import matplotlib
from skimage.metrics import structural_similarity as ssim
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'original_repo'))
from depth_anything_v2.dpt import DepthAnythingV2

# Коды состояний
TEST_SUCCESS = 0
TEST_EXPECTED_FAIL = 1
TEST_CRITICAL_ERROR = 2

def save_heatmap(raw_image, depth_matrix, filename):
    """
    Генерация визуализации как в оригинальном run.py
    """
    # 1. Нормализация глубины
    depth_normalized = (depth_matrix - depth_matrix.min()) / (depth_matrix.max() - depth_matrix.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # 2. Получение colormap 'Spectral_r'
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_color = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    # 3. Создание белой разделительной полосы
    split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    
    # 4. Склейка: оригинал + полоса + карта
    combined_result = cv2.hconcat([raw_image, split_region, depth_color])
    
    save_path = f'output/{filename}'
    cv2.imwrite(save_path, combined_result)
    print(f"  [VIS] Тепловая карта (Spectral_r) сохранена: {save_path}")


def compare_depth_matrices(ref_matrix, test_matrix, test_name):
    """Функция вычитания матриц, расчета метрик и сохранения визуализации"""
    print(f"\n{'='*50}")
    print(f" ЗАПУСК: {test_name}")
    print(f"{'='*50}")
    
    # Проверка размерностей
    if ref_matrix.shape != test_matrix.shape:
        print(f"Несовпадение размеров! Эталон: {ref_matrix.shape}, тест: {test_matrix.shape}. Изменяем размер тестовой карты.")
        test_matrix = cv2.resize(test_matrix, (ref_matrix.shape[1], ref_matrix.shape[0]))

    # Вычитание матриц (абсолютная разность)
    diff_matrix = np.abs(ref_matrix.astype(np.float64) - test_matrix.astype(np.float64)) # ref_matrix - test_matrix
    
    # Нормализация для SSIM
    ref_norm = np.zeros_like(ref_matrix)
    test_norm = np.zeros_like(test_matrix)
    cv2.normalize(ref_matrix, ref_norm, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(test_matrix, test_norm, 0, 1, cv2.NORM_MINMAX)

    # Метрики
    mse_value = np.mean(diff_matrix ** 2)
    ssim_value = ssim(ref_norm, test_norm, data_range=1.0)

    print(f"  [MSE]  Среднеквадратичная ошибка : {mse_value:.8f}")
    print(f"  [SSIM] Структурное сходство      : {ssim_value:.4f} (1.0 = идеал)")

    # Создание карты разности
    os.makedirs('output', exist_ok=True)
    # diff_visual = np.zeros_like(diff_matrix, dtype=np.uint8)
    # cv2.normalize(diff_matrix, diff_visual, 0, 255, cv2.NORM_MINMAX)
    # diff_colored = cv2.applyColorMap(diff_visual, cv2.COLORMAP_JET)
    diff_vis = cv2.normalize(diff_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    safe_filename = test_name.replace(" ", "_").replace(":", "").replace(",", "")
    save_path = f'output/diff_{safe_filename}.png'
    cv2.imwrite(save_path, diff_vis) # diff_colored
    print(f"  [VIS] Карта отклонений сохранена: {save_path}")

    # Оценка результата
    if mse_value < 1e-4 and ssim_value > 0.95:
        print("\n  ✅ РЕЗУЛЬТАТ: ТЕСТ ПРОЙДЕН. Карты идентичны.")
        return TEST_SUCCESS
    else:
        print("\n  ⚠️ РЕЗУЛЬТАТ: ТЕСТ ПРОЙДЕН (ОЖИДАЕМЫЙ ОТКАЗ). Карты глубоко различаются.")
        return TEST_EXPECTED_FAIL


def main():
    print("\n  СИСТЕМА ВАЛИДАЦИИ DOCKER-ОБРАЗА DEPTH-ANYTHING")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nВычислительное устройство: {DEVICE}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vits' 
    
    print("Инициализация модели...")
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    ref_depth = np.load('reference_depth.npy')

    # ТЕСТ 1: Положительный
    img1_path = 'tests/test_image_1.jpg'
    raw_img1 = cv2.imread(img1_path)
    user_depth_1 = model.infer_image(raw_img1)
    
    save_heatmap(raw_img1, user_depth_1, 'heatmap_1_test_image.png')

    status_1 = compare_depth_matrices(ref_depth, user_depth_1, "Test_1_True")

    # ТЕСТ 2: Негативный 
    img2_path = 'tests/test_image_2.jpg'
    raw_img2 = cv2.imread(img2_path)
    user_depth_2 = model.infer_image(raw_img2) # raw_img2_resized
    
    save_heatmap(raw_img2, user_depth_2, 'heatmap_2_test_image.png')

    status_2 = compare_depth_matrices(ref_depth, user_depth_2, "Test_2_False")

    # ИТОГ
    print("\n" + "="*50)
    
    if TEST_CRITICAL_ERROR in [status_1, status_2]:
        print("❌ ИТОГО: КРИТИЧЕСКАЯ ОШИБКА В ЛОГИКЕ ТЕСТОВ!")
        print("Сборка контейнера некорректна.")
    elif status_1 == TEST_SUCCESS and status_2 == TEST_EXPECTED_FAIL:
        print("✅ ИТОГО: ВСЕ ТЕСТЫ ОТРАБОТАЛИ КОРРЕКТНО!")
        print("Модель отличает правильную карту от неправильной.")
    else:
        print("⚠️ ИТОГО: НЕОЖИДАННЫЙ РЕЗУЛЬТАТ ТЕСТИРОВАНИЯ.")
        

if __name__ == '__main__':
    main()