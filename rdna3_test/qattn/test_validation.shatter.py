import torch
import torch.nn.functional as F
import attn_sm80


def test_shatter_bounds():
    device = "cuda"
    B, H, S, D = 2, 4, 1013, 128  # 1013 is our nasty, unaligned prime number

    print(f"--- Running Shatter Test (S={S}, Causal=True) ---")

    # 1. Generate float references (Layout 1: B, H, S, D)
    q_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    k_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    v_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    sm_scale = 1.0 / (D**0.5)

    # 2. Emulate Int8 (Values stay safely within [-128, 127])
    q_int8 = (q_ref * 64).to(torch.int8)
    k_int8 = (k_ref * 64).to(torch.int8)

    # 3. Dynamic Scale Shapes for kPerWarp (so it never crashes on weird S values)
    q_scale_len = ((S + 127) // 128) * 4  # div_ceil(S, 128) * (128/32)
    k_scale_len = ((S + 63) // 64) * 1  # div_ceil(S, 64) * (64/64)
    qs = torch.ones((B, H, q_scale_len), dtype=torch.float32, device=device)
    ks = torch.ones((B, H, k_scale_len), dtype=torch.float32, device=device)

    o = torch.zeros_like(v_ref)

    # 4. PyTorch Ground Truth (MUST have is_causal=True)
    with torch.autocast("cuda", enabled=False):
        expected = F.scaled_dot_product_attention(
            q_ref.float(), k_ref.float(), v_ref.float(), scale=sm_scale, is_causal=True
        ).half()

    # 5. RDNA 3 Kernel Call
    attn_sm80.qk_int8_sv_f16_accum_f32_attn(
        q_int8.contiguous(),
        k_int8.contiguous(),
        v_ref.contiguous(),
        o,
        qs,
        ks,
        1,  # tensor_layout = 1 for [B, H, S, D]
        1,  # is_causal = 1 (True)
        2,  # qk_quant_gran = 2 (PerWarp)
        sm_scale,
        0,  # return_lse = 0
    )

    # 6. Compare
    diff = (expected - o).abs()
    print(f"  Mean Absolute Error: {diff.mean().item():.6f}")
    print(f"  Max Absolute Error:  {diff.max().item():.6f}")

    if torch.isnan(o).any():
        print("  ❌ FAILED: NaN detected. Out-of-bounds memory was read.")
    elif diff.mean().item() > 0.05:
        print("  ❌ FAILED: Math is corrupted at the boundaries.")
    else:
        print("  ✅ PASSED: The RDNA 3 bounds hold!")


if __name__ == "__main__":
    test_shatter_bounds()
