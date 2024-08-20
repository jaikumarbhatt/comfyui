from numba import cuda
import gc
# from inferenceserver import loadmodels, deleteloadedmodels
import os
import torch
import time
import GPUtil
gpus = GPUtil.getGPUs()
print(gpus)
for gpu in gpus:
        print(f"GPU ID: {gpu.id}")
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Load: {gpu.load * 100:.2f}%")
        print(f"GPU Memory Free: {gpu.memoryFree:.2f} MB")
        print(f"GPU Memory Used: {gpu.memoryUsed:.2f} MB")
        print(f"GPU Memory Total: {gpu.memoryTotal:.2f} MB")
        print(f"GPU Temperature: {gpu.temperature:.2f} °C")
        print(f"GPU Driver Version: {gpu.driver}\n")
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"


# loadmodels()
# # deleteloadedmodels()
# # gc.collect()
# loadmodels()
torch.cuda.init()
memory_allocated = torch.cuda.memory_allocated(0)
memory_reserved = torch.cuda.memory_reserved(0)
memory_summary = torch.cuda.memory_summary(device=torch.device("cuda:0"), abbreviated=False)

# Print GPU information
# print(f"GPU Name: {gpu_name}")
print(f"Memory Allocated: {memory_allocated / 1e6:.2f} MB")
print(f"Memory Reserved: {memory_reserved / 1e6:.2f} MB")
print("Memory Summary:")
print(memory_summary)

