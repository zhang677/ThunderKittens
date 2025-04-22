import sys
import test_04_15_09_58
import torch

torch.manual_seed(42)
B, H, N = 13, 8, 512
for M in [16, 32]:
    for D in [64, 128]:
        q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
        k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
        v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
        ref_output = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous(), is_causal=False).transpose(1,2).contiguous()
        output = torch.zeros_like(q)
        if M == 32 and D == 64:
            test_04_15_09_58.wrapped_attend_ker_32_64(q, k, v, output)
        elif M == 16 and D == 64:
            test_04_15_09_58.wrapped_attend_ker_16_64(q, k, v, output)
        elif M == 32 and D == 128:
            test_04_15_09_58.wrapped_attend_ker_32_128(q, k, v, output)
        elif M == 16 and D == 128:
            test_04_15_09_58.wrapped_attend_ker_16_128(q, k, v, output)
        else:
            print("Unsupported configuration")
            continue
        assert torch.allclose(ref_output, output, atol=1e-2), "Outputs are not close!"
        print(f"Test passed for M={M}, D={D}")