import sys
import os
cur_path = "/workspace/ThunderKittens/kernels/test_04_15_09_58"
print(cur_path)
sys.path.append(cur_path) 

import test_04_15_09_58
import torch
import torch.nn.functional as F


torch.manual_seed(42)
B, H, M, N, D = 13, 8, 16, 512, 128
q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
# is_causal = False
# # Enable FlashAttention and disable other implementations
# with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
#     ref_output = F.scaled_dot_product_attention(
#         q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
#         is_causal=is_causal    # Optional
#     )

output = torch.zeros_like(q)
test_04_15_09_58.wrapped_attend_ker(q, k, v, output)
print(output)