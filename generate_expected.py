import os
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIG = {
    'encoder': 'vits',
    'features': 64,
    'out_channels': [48, 96, 192, 384]
}

CHECKPOINT_PATH = 'checkpoints/depth_anything_v2_vits.pth'
TEST_IMAGE_PATH = 'tests/test_image.jpg'
OUTPUT_PATH = 'tests/expected_depth.npy'

# Отладочная информация
abs_test_path = os.path.abspath(TEST_IMAGE_PATH)
print(f"Ожидаемый путь к изображению: {abs_test_path}")
print(f"Файл существует: {os.path.exists(abs_test_path)}")

# Загружаем модель
model = DepthAnythingV2(**MODEL_CONFIG)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()

# Загружаем изображение
raw_img = cv2.imread(TEST_IMAGE_PATH)
if raw_img is None or not isinstance(raw_img, np.ndarray):
    raise ValueError(f"Не удалось загрузить изображение из {TEST_IMAGE_PATH}")

print(f"Изображение загружено, размер: {raw_img.shape}")

# Инференс: передаём исходный BGR-массив
with torch.no_grad():
    depth = model.infer_image(raw_img)

# Сохраняем эталон
np.save(OUTPUT_PATH, depth)
print(f"Эталонная глубина сохранена в {OUTPUT_PATH}")