import cv2
import torch
import fastcv
import numpy as np

img = cv2.imread("fastcv/artifacts/diagonal_gradient.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
lut_numpy = np.arange(256, dtype=np.uint8)
lut_numpy = 255 - lut_numpy
lut_tensor = torch.from_numpy(lut_numpy).cuda()
result_tensor = fastcv.lut(img_tensor, lut_tensor)
result_np = result_tensor.cpu().numpy()
cv2.imwrite("output_lut.jpg", result_np)

print("saved lut image.")