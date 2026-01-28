import cv2
import torch
import fastcv
import numpy as np
import time


def benchmark_lut(img_path="artifacts/diagonal_gradient.jpg", runs=50):

    img_np = cv2.imread(img_path)

    #tworzenie tabeli i invertowanie jej 
    lut_np = np.arange(256, dtype=np.uint8)
    lut_np = 255 - lut_np

    img_torch = torch.from_numpy(img_np).cuda()
    lut_torch = torch.from_numpy(lut_np).cuda()

    #procek
    start = time.perf_counter()

    for _ in range(runs):
        _ = cv2.LUT(img_np, lut_np)
    end = time.perf_counter()

    milis_in_sek = 1000
    cpu_time = (end - start) / runs * milis_in_sek


    #gpu
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        _ = fastcv.lut(img_torch, lut_torch)
    torch.cuda.synchronize()
    end = time.perf_counter()

    gpu_time = (end - start) / runs * milis_in_sek



    print(f"OpenCV (CPU): {cpu_time:.4f} ms | fastcv (CUDA): {gpu_time:.4f} ms")

    return cpu_time, gpu_time


if __name__ == "__main__":
    cpu_time, gpu_time = benchmark_lut()
    print("\n=== Final Results ===")
    print("OpenCV (CPU)\tfastcv (CUDA)")
    print(f"{cpu_time:.4f} ms\t{gpu_time:.4f} ms")
