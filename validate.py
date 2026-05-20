import os
import sys
import numpy as np
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from skimage.metrics import structural_similarity as ssim
import argparse

def main():
    # Проверка доступных устройств
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Используемое устройство: {device}")

    # Конфигурация модели
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
       # 'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
       # 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
       # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = os.environ.get('ENCODER', 'vits')  # можно переопределить через переменную окружения
    print(f"Используемый энкодер: {encoder}")

    # Пути к тестовым данным (делаем параметрами)
    test_image_path = os.environ.get('TEST_IMAGE', 'tests/test_image_1.jpg')
    reference_depth_path = os.environ.get('REF_DEPTH', 'reference_depth.npy')
    output_diff_path = os.environ.get('DIFF_OUTPUT', 'difference_map.png')

    # Проверяем наличие чекпоинта
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    if not os.path.isfile(checkpoint_path):
        print(f"Ошибка: чекпоинт не найден по пути {checkpoint_path}")
        sys.exit(1)

    # Загрузка модели
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device).eval()
    print("Модель загружена успешно.")

    # Чтение тестового изображения
    raw_img = cv2.imread(test_image_path)
    if raw_img is None:
        print(f"Ошибка: не удалось загрузить тестовое изображение {test_image_path}")
        sys.exit(1)

    # Инференс
    with torch.no_grad():
        depth_test = model.infer_image(raw_img)   # HxW numpy array 
    print(f"Размер предсказанной карты глубины: {depth_test.shape}, dtype={depth_test.dtype}")

    # Загрузка эталонной карты
    if not os.path.isfile(reference_depth_path):
        print(f"Ошибка: эталонная карта не найдена: {reference_depth_path}")
        sys.exit(1)
    depth_ref = np.load(reference_depth_path)
    print(f"Размер эталонной карты глубины: {depth_ref.shape}, dtype={depth_ref.dtype}")

    # Проверка совпадения размеров
    if depth_ref.shape != depth_test.shape:
        print(f"Несовпадение размеров! Эталон: {depth_ref.shape}, тест: {depth_test.shape}. Изменяем размер тестовой карты.")
        depth_test = cv2.resize(depth_test, (depth_ref.shape[1], depth_ref.shape[0]))

    # Сравнение
    # 1) Вычитание матриц (абсолютная разность)
    diff = np.abs(depth_ref.astype(np.float64) - depth_test.astype(np.float64))
    print(f"Максимальная абсолютная разность: {diff.max():.6f}")
    print(f"Средняя абсолютная разность: {diff.mean():.6f}")

    # 2) MSE
    mse = np.mean((depth_ref.astype(np.float64) - depth_test.astype(np.float64)) ** 2)
    print(f"MSE: {mse:.10f}")

    # 3) SSIM (работает с изображениями в диапазоне 0..1, нужно нормировать)
    # Нормируем обе карты в диапазон [0,1] относительно их максимальных значений
    ref_norm = depth_ref.astype(np.float64) / depth_ref.max()
    test_norm = depth_test.astype(np.float64) / depth_test.max()
    ssim_value, ssim_map = ssim(ref_norm, test_norm, full=True, data_range=1.0)
    print(f"SSIM: {ssim_value:.6f}")

    # Сохраняем карту разности как изображение (нормируем для визуализации)
    diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(output_diff_path, diff_vis)
    print(f"Карта разности сохранена в {output_diff_path}")

    # Оценка результата
    if mse < 1e-6 and ssim_value > 0.999:
        print("\n✅ ТЕСТ ПРОЙДЕН: карты глубины практически идентичны.")
    else:
        print("\n❌ ТЕСТ НЕ ПРОЙДЕН: обнаружены расхождения. Проверьте окружение или версию модели.")

if __name__ == '__main__':
    main()