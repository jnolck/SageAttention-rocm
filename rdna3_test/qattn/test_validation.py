import torch

# import attn_sm80
from sageattention import attn_sm80

print("=== SageAttention RDNA 3 Validation ===")

# 1. Setup Dimensions (Small enough to debug, large enough to use warps)
B, N, H, D = 1, 128, 1, 64
sm_scale = 1.0 / (D**0.5)  # Standard attention scaling

# 2. Generate Data
# Keep integers small to avoid extreme exponential explosion in softmax
q = torch.randint(-3, 3, (B, N, H, D), dtype=torch.int8, device="cuda")
k = torch.randint(-3, 3, (B, N, H, D), dtype=torch.int8, device="cuda")
v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")

# 3. Setup Scales (Set to 1.0 to make the Python reference math simple)
q_scale = torch.ones((B, H, 4), dtype=torch.float32, device="cuda")
k_scale = torch.ones((B, H, 2), dtype=torch.float32, device="cuda")
o_custom = torch.empty((B, N, H, D), dtype=torch.float16, device="cuda")

# ==========================================
# 🥊 ROUND 1: PyTorch Native Reference Math
# ==========================================
# PyTorch expects [Batch, Heads, SeqLen, HeadDim] for standard matrix math
q_ref = q.transpose(1, 2).float()
k_ref = k.transpose(1, 2).float()
v_ref = v.transpose(1, 2).float()

# Attention Formula: Softmax(Q * K^T * scale) * V
scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * sm_scale
probs = torch.softmax(scores, dim=-1)
out_ref_transposed = torch.matmul(probs, v_ref)

# Bring back to [B, N, H, D] and cast to FP16
out_ref = out_ref_transposed.transpose(1, 2).to(torch.float16)


# ==========================================
# 🥊 ROUND 2: Your Custom RDNA 3 Kernel
# ==========================================
attn_sm80.qk_int8_sv_f16_accum_f16_attn(
    q,
    k,
    v,
    o_custom,
    q_scale,
    k_scale,
    0,  # tensor_layout (0 = [B, N, H, D])
    0,  # is_causal (0 = False)
    2,  # qk_quant_gran (2 = kPerWarp)
    sm_scale,
    0,  # return_lse
)

# ==========================================
# ⚖️ THE VERDICT
# ==========================================
# Calculate the absolute difference between the two outputs
max_error = torch.max(torch.abs(out_ref - o_custom)).item()
mean_error = torch.mean(torch.abs(out_ref - o_custom)).item()

print(f"\nMax Absolute Error:  {max_error:.6f}")
print(f"Mean Absolute Error: {mean_error:.6f}")

if max_error < 0.05:  # Allowing a tiny margin for FP16 rounding differences
    print("\n✅ SUCCESS! THE KERNEL IS MATHEMATICALLY PERFECT.")
else:
    print("\n❌ MISMATCH! We have a matrix alignment or layout issue.")
    print("Reference first 4:", out_ref[0, 0, 0, :4].tolist())
    print("Custom first 4:   ", o_custom[0, 0, 0, :4].tolist())
