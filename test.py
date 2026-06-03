import torch

# 1. Check if CUDA is available
print(torch.cuda.is_available())  
# Should output: True

# 2. Get the number of available GPUs
print(torch.cuda.device_count())  

# 3. Get the name of your GPU
print(torch.cuda.get_device_name(0))  
