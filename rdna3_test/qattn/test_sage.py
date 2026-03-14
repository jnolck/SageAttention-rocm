import torch
import attn_sm80

print(">>> Booting Pure RDNA 3 Bare-Metal Test...")

B, S, H, D = 2, 4096, 10, 64

print(">>> Allocating raw tensors for C++...")

# 1. We MUST allocate the physical INT8 memory for Q and K
q_int8 = torch.randint(-127, 127, (B, S, H, D), dtype=torch.int8, device="cuda")
k_int8 = torch.randint(-127, 127, (B, S, H, D), dtype=torch.int8, device="cuda")

# 2. Value remains FP16
v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

# 3. Pre-allocate the empty output tensor for C++ to write into
out = torch.empty((B, S, H, D), dtype=torch.float16, device="cuda")

# 4. Corrected scales (q_scale is 128, k_scale is 64 to satisfy the bare-metal asserts)
q_scale = torch.ones((B, H, 128), dtype=torch.float32, device="cuda")
k_scale = torch.ones((B, H, 64), dtype=torch.float32, device="cuda")

torch.cuda.synchronize()
print(">>> Firing the raw f32 accumulator kernel...")

try:
    # Pass ALL 11 arguments exactly as fused.h demands
    # attn_sm80.qk_int8_sv_f16_accum_f32_attn(
    #     q_int8,  # 1. Query (int8)
    #     k_int8,  # 2. Key (int8)
    #     v,  # 3. Value (fp16)
    #     out,  # 4. Output buffer (fp16)
    #     q_scale,  # 5. Query scales (fp32)
    #     k_scale,  # 6. Key scales (fp32)
    #     0,  # 7. tensor_layout (0 = [B, S, H, D])
    #     0,  # 8. is_causal (0 = False)
    #     2,  # 9. qk_quant_gran (2 = PerWarp)
    #     1.0,  # 10. sm_scale (float)
    #     0,  # 11. return_lse (0 = False)
    # )
    attn_sm80.qk_int8_sv_f16_accum_f16_attn(
        q_int8, k_int8, v, out, q_scale, k_scale, 0, 0, 2, 1.0, 0
    )

    torch.cuda.synchronize()
    print("\n=======================================================")
    print("🚀 HELL YES. BARE METAL MATH COMPLETED. 🚀")
    print(f"Output Tensor Shape: {out.shape}")
    print("=======================================================\n")

except Exception as e:
    print(f"\n❌ FATAL KERNEL ERROR:\n{e}\n")
