import sys
import os
import test_04_15_09_58
import torch

torch.manual_seed(42)
B, H, N = 13, 8, 512

if len(sys.argv) != 3:
    print("Usage: python bench.py <M> <D>")
    sys.exit(1)
else:
    M = int(sys.argv[1])
    D = int(sys.argv[2])

q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)

output = torch.zeros_like(q)
if M == 16 and D == 64:
    test_04_15_09_58.wrapped_attend_ker_16_64(q, k, v, output)
elif M == 16 and D == 96:
    test_04_15_09_58.wrapped_attend_ker_16_96(q, k, v, output)
elif M == 16 and D == 128:
    test_04_15_09_58.wrapped_attend_ker_16_128(q, k, v, output)
elif M == 16 and D == 160:
    test_04_15_09_58.wrapped_attend_ker_16_160(q, k, v, output)
elif M == 32 and D == 64:
    test_04_15_09_58.wrapped_attend_ker_32_64(q, k, v, output)
elif M == 32 and D == 96:
    test_04_15_09_58.wrapped_attend_ker_32_96(q, k, v, output)
elif M == 32 and D == 128:
    test_04_15_09_58.wrapped_attend_ker_32_128(q, k, v, output)
elif M == 32 and D == 160:
    test_04_15_09_58.wrapped_attend_ker_32_160(q, k, v, output)
elif M == 48 and D == 64:
    test_04_15_09_58.wrapped_attend_ker_48_64(q, k, v, output)
elif M == 48 and D == 96:
    test_04_15_09_58.wrapped_attend_ker_48_96(q, k, v, output)
elif M == 48 and D == 128:
    test_04_15_09_58.wrapped_attend_ker_48_128(q, k, v, output)
elif M == 48 and D == 160:
    test_04_15_09_58.wrapped_attend_ker_48_160(q, k, v, output)
elif M == 64 and D == 64:
    test_04_15_09_58.wrapped_attend_ker_64_64(q, k, v, output)
elif M == 64 and D == 96:
    test_04_15_09_58.wrapped_attend_ker_64_96(q, k, v, output)
elif M == 64 and D == 128:
    test_04_15_09_58.wrapped_attend_ker_64_128(q, k, v, output)
elif M == 64 and D == 160:
    test_04_15_09_58.wrapped_attend_ker_64_160(q, k, v, output)
else:
    print("Unsupported configuration")
    sys.exit(1)
