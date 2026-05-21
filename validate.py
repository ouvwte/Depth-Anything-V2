import cv2
import torch
import numpy as np
import sys
import os

# Путь к оригинальному коду
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'original_repo'))

from depth_anything_v2.dpt import DepthAnythingV2

def main():
    print("--- ЗАПУСК ВАЛИДАЦИИ DOCKER-ОБРАЗА ---")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство для инференса: {DEVICE}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vits' 
    
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    print("Модель загружена.")

    test_img_path = 'tests/test_image.jpg'
    raw_img = cv2.imread(test_img_path)
    if raw_img is None:
        raise FileNotFoundError(f"Изображение {test_img_path} не найдено!")
        
    user_depth = model.infer_image(raw_img)
    print(f"Инференс завершен. Формат: {user_depth.dtype}")

    ref_depth = np.load('reference_depth.npy')

    if ref_depth.shape != user_depth.shape:
        print(f"❌ ТЕСТ ПРОВАЛЕН: Размерности не совпадают!")
        return False

    # МАТРИЧНОЕ ВЫЧИТАНИЕ
    diff_matrix = ref_depth - user_depth
    
    # MSE
    mse_value = np.mean(diff_matrix ** 2)
    max_abs_diff = np.max(np.abs(diff_matrix))
    
    print("\n--- Результаты ---")
    print(f"Макс. разница: {max_abs_diff:.6f}")
    print(f"MSE: {mse_value:.8f}")

    if mse_value < 1e-4:
        print("\n✅ ТЕСТ ПРОЙДЕН!")
        return True
    else:
        print("\n❌ ТЕСТ ПРОВАЛЕН!")
        return False

if __name__ == '__main__':
    main()