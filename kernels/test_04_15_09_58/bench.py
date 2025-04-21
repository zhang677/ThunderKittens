import sys
import os
cur_path = "/workspace/ThunderKittens/kernels/test_04_15_09_58"
print(cur_path)
sys.path.append(cur_path) 

import test_04_15_09_58
import torch


torch.manual_seed(42)
B, H, M, N, D = 13, 8, 16, 512, 128
q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)

output = torch.zeros_like(q)
test_04_15_09_58.wrapped_attend_ker(q, k, v, output)
