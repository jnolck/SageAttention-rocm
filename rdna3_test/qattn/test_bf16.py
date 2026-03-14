import torch
from sageattention import attn_sm80
from sageattention import sage_fused_quant


def test_bf16_pipeline():
    B = 1
    H_Q = 4
    H_KV = 4
    T_Q = 128
    T_KV = 64  # Changed to 64 so block_size 64 perfectly matches 1 scale factor
    D = 64

    print(">>> 1. Allocating FP16 Source Tensors (V must be FP16)...")
    query_fp16 = torch.randn((B, H_Q, T_Q, D), dtype=torch.float16, device="cuda")
    key_fp16 = torch.randn((B, H_KV, T_KV, D), dtype=torch.float16, device="cuda")
    value_fp16 = torch.randn((B, H_KV, T_KV, D), dtype=torch.float16, device="cuda")
    value_mean = torch.randn((B, H_KV, D), dtype=torch.bfloat16, device="cuda")

    print(">>> 2. Quantizing Q and K to INT8...")
    query_int8 = torch.empty((B, H_Q, T_Q, D), dtype=torch.int8, device="cuda")
    key_int8 = torch.empty((B, H_KV, T_KV, D), dtype=torch.int8, device="cuda")

    query_scale = torch.empty((B, H_Q, 4), dtype=torch.float32, device="cuda")
    key_scale = torch.empty((B, H_KV, 1), dtype=torch.float32, device="cuda")

    # Fire the armory
    sage_fused_quant.quant_per_warp_int8_cuda(
        query_fp16, query_int8, query_scale, 128, 32, 1
    )
    sage_fused_quant.quant_per_block_int8_cuda(key_fp16, key_int8, key_scale, 64, 1)

    print(">>> 3. Allocating BFLOAT16 Output Buffer...")
    # THIS IS THE CRITICAL TEST - Forcing the kernel to use the BF16 conversion logic
    output_bf16 = torch.empty((B, H_Q, T_Q, D), dtype=torch.bfloat16, device="cuda")

    print(">>> 4. Firing RDNA 3 Fuse V-Mean Attention Kernel...")
    try:
        attn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
            query_int8,
            key_int8,
            value_fp16,
            output_bf16,
            query_scale,
            key_scale,
            value_mean,
            1,
            0,
            2,
            1.0,
            0,
        )
        torch.cuda.synchronize()
        print("✅ BFLOAT16 kernel executed successfully!")
        print(f"✅ Final Output shape: {output_bf16.shape}")
        print(f"✅ Final Output dtype: {output_bf16.dtype}")
    except Exception as e:
        print(f"❌ Attention Failed: {e}")


if __name__ == "__main__":
    test_bf16_pipeline()
