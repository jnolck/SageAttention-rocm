import torch
import attn_sm80

# 1. Define tiny dimensions just to test the plumbing
B, N, H, D = 1, 128, 1, 64

# 2. Allocate the exact tensor layouts the C++ checks expect
# tensor_layout=0 means [Batch, SeqLen, Heads, HeadDim]
q = torch.randint(-5, 5, (B, N, H, D), dtype=torch.int8, device="cuda")
k = torch.randint(-5, 5, (B, N, H, D), dtype=torch.int8, device="cuda")
v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
o = torch.empty((B, N, H, D), dtype=torch.float16, device="cuda")

# 3. Setup the quantization scales
# Based on CTA_Q=128, WARP_Q=32 -> 4 warps per block
q_scale = torch.ones((B, H, 4), dtype=torch.float32, device="cuda")
# Based on CTA_K=64, WARP_K=64 -> 1 warp per block. N=128 needs 2 blocks.
k_scale = torch.ones((B, H, 2), dtype=torch.float32, device="cuda")

print("[1/3] Tensors allocated in VRAM.")
print("[2/3] Firing RDNA 3 Matrix Cores...")

# 4. The main event
# Args: q, k, v, out, q_scale, k_scale, layout, causal, quant_granularity, sm_scale, return_lse
attn_sm80.qk_int8_sv_f16_accum_f32_attn(
    q,
    k,
    v,
    o,
    q_scale,
    k_scale,
    0,  # tensor_layout (0 = [B, N, H, D])
    0,  # is_causal (0 = False)
    2,  # qk_quant_gran (1 = kPerWarp)
    1.0,  # sm_scale
    0,  # return_lse (0 = False)
)

print("[3/3] Kernel executed safely! No segfaults.")
print("Sample Output (first 4 elements of Output):", o[0, 0, 0, :4].tolist())
