import torch
from . import sage_fused_quant


def quantize_per_warp_int8(q, tensor_layout="NHD"):
    q_c = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    q_c.copy_(q)

    # Let ComfyUI pass NHD [B, S, H, D]
    if tensor_layout == "NHD":
        B, S, H, D = q_c.shape
        layout_int = 0
    else:
        B, H, S, D = q_c.shape
        layout_int = 1

    q_scale_len = ((S + 127) // 128) * 4
    q_int8 = torch.empty(q_c.shape, dtype=torch.int8, device=q_c.device)

    # CRITICAL FIX: The C++ code demands exactly a 3D Tensor shaped [B, H, L]
    # regardless of whether the input is NHD or HND.
    q_scale = torch.empty((B, H, q_scale_len), dtype=torch.float32, device=q_c.device)

    sage_fused_quant.quant_per_warp_int8_cuda(q_c, q_int8, q_scale, 128, 32, layout_int)
    return q_int8, q_scale, q_c


def quantize_per_block_int8(k, tensor_layout="NHD"):
    k_c = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    k_c.copy_(k)

    if tensor_layout == "NHD":
        B, S, H, D = k_c.shape
        layout_int = 0
    else:
        B, H, S, D = k_c.shape
        layout_int = 1

    k_scale_len = ((S + 63) // 64) * 1
    k_int8 = torch.empty(k_c.shape, dtype=torch.int8, device=k_c.device)

    # CRITICAL FIX: The C++ code demands exactly a 3D Tensor shaped [B, H, L]
    k_scale = torch.empty((B, H, k_scale_len), dtype=torch.float32, device=k_c.device)

    sage_fused_quant.quant_per_block_int8_cuda(k_c, k_int8, k_scale, 64, layout_int)
    return k_int8, k_scale, k_c
