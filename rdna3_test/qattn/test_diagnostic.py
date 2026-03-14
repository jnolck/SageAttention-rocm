import torch
import torch.nn.functional as F
import attn_sm80


def run_test(S, is_causal_flag):
    device = "cuda"
    B, H, D = 2, 4, 128

    # Generate float references
    q_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    k_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    v_ref = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    sm_scale = 1.0 / (D**0.5)

    # Emulate Int8 safely (multiplied by 64)
    q_int8 = (q_ref * 64).to(torch.int8)
    k_int8 = (k_ref * 64).to(torch.int8)

    q_scale_len = ((S + 127) // 128) * 4
    k_scale_len = ((S + 63) // 64) * 1

    # FIX: Divide by 64.0 to invert the int8 multiplication!
    qs = torch.ones((B, H, q_scale_len), dtype=torch.float32, device=device) / 64.0
    ks = torch.ones((B, H, k_scale_len), dtype=torch.float32, device=device) / 64.0

    o = torch.zeros_like(v_ref)

    # PyTorch Ground Truth
    with torch.autocast("cuda", enabled=False):
        expected = F.scaled_dot_product_attention(
            q_ref.float(),
            k_ref.float(),
            v_ref.float(),
            scale=sm_scale,
            is_causal=is_causal_flag,
        ).half()

    # RDNA 3 Kernel Call
    attn_sm80.qk_int8_sv_f16_accum_f32_attn(
        q_int8.contiguous(),
        k_int8.contiguous(),
        v_ref.contiguous(),
        o,
        qs,
        ks,
        1,
        1 if is_causal_flag else 0,
        2,
        sm_scale,
        0,
    )

    diff = (expected - o).abs()
    print(
        f"  -> Mean Error: {diff.mean().item():.6f} | Max Error: {diff.max().item():.6f}"
    )


if __name__ == "__main__":
    print("\n[TEST 1] ALIGNED (S=1024), NO CAUSAL")
    run_test(1024, False)

    print("\n[TEST 2] ALIGNED (S=1024), CAUSAL")
    run_test(1024, True)

    print("\n[TEST 3] UNALIGNED (S=1013), NO CAUSAL")
    run_test(1013, False)

    print("\n[TEST 4] UNALIGNED (S=1013), CAUSAL")
    run_test(1013, True)
    print("")
