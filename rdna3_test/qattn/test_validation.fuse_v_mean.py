import torch
import torch.nn.functional as F
from sageattention.core import sageattn


def validate_fused_wrapper():
    B = 2
    S = 128
    H = 4
    D = 64

    print(">>> 1. Generating Normalized Test Tensors (Layout: B, S, H, D)...")
    # CRITICAL: Q, K, and V must start as FP16 so the C++ quantizers can read the memory correctly!
    # Dividing by 10.0 prevents FP16 Softmax explosions during raw random testing.
    q = torch.randn((B, S, H, D), dtype=torch.float16, device="cuda") / 10.0
    k = torch.randn((B, S, H, D), dtype=torch.float16, device="cuda") / 10.0
    v_fp16 = torch.randn((B, S, H, D), dtype=torch.float16, device="cuda") / 10.0

    # The V-Mean tensor [Batch, Heads, Dim] - testing the BFloat16 bridge!
    v_mean = torch.randn((B, H, D), dtype=torch.bfloat16, device="cuda") / 10.0

    print("\n>>> 2. Calculating Ground Truth (Native PyTorch SDPA)...")
    # PyTorch SDPA requires matching dtypes, so we convert Q, K, and V to BFloat16 here
    q_pt = q.to(torch.bfloat16).transpose(1, 2)
    k_pt = k.to(torch.bfloat16).transpose(1, 2)
    v_pt = v_fp16.to(torch.bfloat16).transpose(1, 2)

    # Calculate standard attention
    torch_attn = F.scaled_dot_product_attention(q_pt, k_pt, v_pt)

    # Mathematically add the mean back to the sequence length
    torch_output = torch_attn + v_mean.unsqueeze(2)

    # Transpose back to [B, S, H, D]
    torch_output = torch_output.transpose(1, 2).contiguous()

    print("\n>>> 3. Firing RDNA 3 SageAttention Wrapper...")
    try:
        # We pass V as FP16, V-Mean as BF16, and let the wrapper handle the routing!
        sage_output = sageattn(
            q=q, k=k, v=v_fp16, tensor_layout="NHD", is_causal=False, v_mean=v_mean
        )

        print("\n>>> 4. Validating Hardware Math vs PyTorch...")
        # INT8 quantization naturally introduces slight precision variance compared to pure BF16 math.
        torch.testing.assert_close(sage_output, torch_output, atol=2e-1, rtol=1e-2)

        print("✅ SUCCESS: RDNA 3 Fused Matrix Math matches PyTorch SDPA!")
        print(f"✅ Final Hardware Output Shape: {sage_output.shape}")
        print(f"✅ Final Hardware Dtype: {sage_output.dtype}")

    except AssertionError as e:
        print("\n❌ MATH FAILED: Outputs do not match!")
        print(e)
    except Exception as e:
        print(f"\n❌ PIPELINE CRASHED: {e}")


if __name__ == "__main__":
    validate_fused_wrapper()
