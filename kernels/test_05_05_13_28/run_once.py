import sys
import os
import test_05_05_13_28
import torch


torch.manual_seed(42)

if __name__ == "__main__":  
    if len(sys.argv) != 5:
        print("Usage: python run_once.py B M N D")
        sys.exit(1)
    else:
        B = int(sys.argv[1])
        M = int(sys.argv[2])
        N = int(sys.argv[3])
        D = int(sys.argv[4])

        if M == 16 and D == 64:
            H = 12
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_64_12(q, k, v, output)
        elif M == 16 and D == 96:
            H = 8
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_96_8(q, k, v, output)
        elif M == 16 and D == 128:
            H = 7
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_128_7(q, k, v, output)
        elif M == 16 and D == 160:
            H = 4
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_160_4(q, k, v, output)
        elif M == 32 and D == 64:
            H = 8
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_64_8(q, k, v, output)
        elif M == 32 and D == 96:
            H = 8
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_96_8(q, k, v, output)
        elif M == 32 and D == 128:
            H = 6
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_128_6(q, k, v, output)
        elif M == 32 and D == 160:
            H = 5
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_160_5(q, k, v, output)
        elif M == 48 and D == 64:
            H = 8
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_64_8(q, k, v, output)
        elif M == 48 and D == 96:
            H = 7
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_96_7(q, k, v, output)
        elif M == 48 and D == 128:
            H = 5
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_128_5(q, k, v, output)
        elif M == 48 and D == 160:
            H = 4
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_160_4(q, k, v, output)
        elif M == 64 and D == 64:
            H = 8
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_64_8(q, k, v, output)
        elif M == 64 and D == 96:
            H = 6
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_96_6(q, k, v, output)
        elif M == 64 and D == 128:
            H = 4
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_128_4(q, k, v, output)
        elif M == 64 and D == 160:
            H = 3
            q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_160_3(q, k, v, output)
        else:
            print("Unsupported configuration")