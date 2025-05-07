import test_05_05_13_28
import torch

torch.manual_seed(42)

def get_ref_output(B, H, M, N, D):
    q = torch.randn((B, M, H, D), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
    v = torch.randn((B, N, H, D), device='cuda', dtype=torch.bfloat16)
    ref_output = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous(), is_causal=False).transpose(1,2).contiguous()
    return ref_output, q, k, v

B, N = 108, 512
for M in [16, 32, 48, 64]:
    for D in [64, 96, 128, 160]:
        if M == 16 and D == 64:
            H = 12
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_64_12(q, k, v, output)
        elif M == 16 and D == 96:
            H = 8
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_96_8(q, k, v, output)
        elif M == 16 and D == 128:
            H = 7
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_128_7(q, k, v, output)
        elif M == 16 and D == 160:
            H = 4
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_16_160_4(q, k, v, output)
        elif M == 32 and D == 64:
            H = 8
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_64_8(q, k, v, output)
        elif M == 32 and D == 96:
            H = 8
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_96_8(q, k, v, output)
        elif M == 32 and D == 128:
            H = 6
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_128_6(q, k, v, output)
        elif M == 32 and D == 160:
            H = 5
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_32_160_5(q, k, v, output)
        elif M == 48 and D == 64:
            H = 8
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_64_8(q, k, v, output)
        elif M == 48 and D == 96:
            H = 7
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_96_7(q, k, v, output)
        elif M == 48 and D == 128:
            H = 5
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_128_5(q, k, v, output)
        elif M == 48 and D == 160:
            H = 4
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_48_160_4(q, k, v, output)
        elif M == 64 and D == 64:
            H = 8
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_64_8(q, k, v, output)
        elif M == 64 and D == 96:
            H = 6
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_96_6(q, k, v, output)
        elif M == 64 and D == 128:
            H = 4
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_128_4(q, k, v, output)
        elif M == 64 and D == 160:
            H = 3
            ref_output, q, k, v = get_ref_output(B, H, M, N, D)
            output = torch.zeros_like(q)
            test_05_05_13_28.wrapped_attend_ker_64_160_3(q, k, v, output)
        else:
            print("Unsupported configuration")
            continue
        assert torch.allclose(ref_output, output, atol=1e-2), f"Outputs are not close for {B}x{H}x{M}x{N}x{D}!"
        print(f"Test passed for {B}x{H}x{M}x{N}x{D}")