import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
  #  'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
  #  'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}  
}

encoder = "vits" # vits, vitb, vitl

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('C:/Users/Daniil/temp/rsgo/practice/zachet/Depth-Anything-V2/tests/test_image.jpg') # your/image/path
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

np.save('reference_depth.npy', depth)
print(f"Эталон сохранен. Формат: {depth.dtype}, Размер: {depth.shape}, Мин/Макс: {depth.min()}/{depth.max()}")