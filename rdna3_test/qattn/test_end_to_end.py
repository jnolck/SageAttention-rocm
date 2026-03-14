import torch
import attn_sm80
import sage_fused_quant  # Your previously compiled quantization library!


def test_end_to_end_fuse():
    # Standard dimensions
    B = 1
    H_Q = 4
    H_KV = 4
    T_Q = 128  # Matches CTA_Q of 128
    T_KV = 128  # Matches CTA_K of 64
    D = 64  # Head dim

    print(">>> 1. Allocating FP16 Source Tensors...")
    query_fp16 = torch.randn((B, H_Q, T_Q, D), dtype=torch.float16, device="cuda")
    key_fp16 = torch.randn((B, H_KV, T_KV, D), dtype=torch.float16, device="cuda")

    # The V tensor stays in FP16 for this kernel
    value_fp16 = torch.randn((B, H_KV, T_KV, D), dtype=torch.float16, device="cuda")

    # The newly unlocked value_mean tensor! Shape: [B, H_KV, D]
    value_mean = torch.randn((B, H_KV, D), dtype=torch.float16, device="cuda")

    print(">>> 2. Allocating INT8 & Scale Destination Tensors...")
    query_int8 = torch.empty((B, H_Q, T_Q, D), dtype=torch.int8, device="cuda")
    key_int8 = torch.empty((B, H_KV, T_KV, D), dtype=torch.int8, device="cuda")

    # Scale shapes for kPerWarp (QuantGranularity = 2)
    query_scale = torch.empty((B, H_Q, 4), dtype=torch.float32, device="cuda")
    key_scale = torch.empty((B, H_KV, 2), dtype=torch.float32, device="cuda")

    print(">>> 3. Firing RDNA 3 Quantizers (FP16 -> INT8 + Scales)...")
    try:
        # The trailing arguments (1, 127.0, 1) usually represent:
        # tensor_layout, quant_max, return_type based on the fused.hip signatures.
        # Adjust if pybind throws an argument error!
        sage_fused_quant.quant_per_warp_int8_cuda(
            query_fp16, query_int8, query_scale, 128, 32, 1
        )
        sage_fused_quant.quant_per_block_int8_cuda(key_fp16, key_int8, key_scale, 64, 1)
        print("✅ Quantization complete!")
    except Exception as e:
        print(f"❌ Quantization Failed: {e}")
        return

    print("\n>>> 4. Allocating Final Output Buffer...")
    output = torch.empty((B, H_Q, T_Q, D), dtype=torch.float16, device="cuda")

    print(">>> 5. Firing RDNA 3 Fuse V-Mean Attention Kernel...")
    try:
        # args: query, key, value, output, q_scale, k_scale, v_mean,
        #       layout, is_causal, quant_gran, sm_scale, return_lse
        attn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
            query_int8,
            key_int8,
            value_fp16,
            output,
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
        print("✅ fused_v_mean kernel executed successfully without Segfault!")
        print(f"✅ Final Output shape verified: {output.shape}")
    except Exception as e:
        print(f"❌ Attention Failed: {e}")


if __name__ == "__main__":
    test_end_to_end_fuse()
