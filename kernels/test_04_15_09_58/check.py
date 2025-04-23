import sys
import test_04_15_09_58
import torch

torch.manual_seed(42)
B, H, N = 13, 8, 512
for M in [16, 32, 48]:
    for D in [64, 96, 128, 160]:
        q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
        k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
        v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
        ref_output = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous(), is_causal=False).transpose(1,2).contiguous()
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
            continue
        assert torch.allclose(ref_output, output, atol=1e-2), "Outputs are not close!"
        print(f"Test passed for M={M}, D={D}")