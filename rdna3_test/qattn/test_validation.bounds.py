import torch
import torch.amp
import torch.nn.functional as F
import attn_sm80


def unaligned_probe():
    device = "cuda"
    B, H, S, D = 2, 4, 111, 128  # S=111 is deliberately nasty

    q_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    k_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    v_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    sm_scale = 1.0 / (D**0.5)

    # Ground truth reference
    with torch.autocast("cuda", enabled=False):
        expected = F.scaled_dot_product_attention(
            q_ref.float(), k_ref.float(), v_ref.float(), scale=sm_scale
        ).half()

    # Emulate int8 ranges
    q_int8 = (q_ref * 64).to(torch.int8)
    k_int8 = (k_ref * 64).to(torch.int8)

    qs = torch.ones((B, H, 4), dtype=torch.float32, device="cuda")
    ks = torch.ones((B, H, 2), dtype=torch.float32, device="cuda")
    o = torch.zeros_like(v_ref)

    print(f"\n--- Running Unaligned Bounds-Check Probe (S={S}) ---")
    try:
        attn_sm80.qk_int8_sv_f16_accum_f32_attn(
            q_int8.contiguous(),
            k_int8.contiguous(),
            v_ref.contiguous(),
            o,
            qs,
            ks,
            1,
            1,
            2,
            sm_scale,
            0,
        )

        diff = (expected - o).abs()
        print(f"  Mean Absolute Error: {diff.mean().item():.6f}")
        print(f"  Max Absolute Error:  {diff.max().item():.6f}")

        if torch.isnan(o).any():
            print("  ❌ FAILED: NaN detected. Out-of-bounds memory was read.")
        elif diff.mean().item() > 0.05:
            print("  ❌ FAILED: Math is corrupted at the boundaries.")
        else:
            print("  ✅ PASSED: No segfault, bounds are secure.")

    except Exception as e:
        print(f"  ❌ CRASH: Boundary check triggered a fatal error! {e}")


if __name__ == "__main__":
    unaligned_probe()
