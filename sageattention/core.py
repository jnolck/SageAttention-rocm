import torch
import torch.nn.functional as F
import traceback
from .quant import quantize_per_warp_int8, quantize_per_block_int8
from .backend_wmma import run_fused_wmma


def sageattn(
    q,
    k,
    v,
    attn_mask=None,
    is_causal=False,
    sm_scale=None,
    use_inst_buf=False,
    **kwargs,
):
    tensor_layout = kwargs.get("tensor_layout", "NHD")

    try:
        _ = q.data_ptr()
        _ = k.data_ptr()
        _ = v.data_ptr()
        has_storage = True
    except RuntimeError:
        has_storage = False

    # Let ComfyUI pass NHD natively.
    if tensor_layout == "NHD":
        S_q, S_k, D = q.shape[1], k.shape[1], q.shape[3]
    else:
        S_q, S_k, D = q.shape[2], k.shape[2], q.shape[3]

    if (
        not has_storage
        or not q.is_cuda
        or not k.is_cuda
        or not v.is_cuda
        or q.numel() == 0
        or k.numel() == 0
        or v.numel() == 0
        or S_q != S_k
        or D not in [64, 128]
        or q.dtype not in [torch.float16, torch.bfloat16]
    ):
        try:
            if tensor_layout == "HND":
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=is_causal
                )
            else:
                q_p, k_p, v_p = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q_p, k_p, v_p, attn_mask=attn_mask, is_causal=is_causal
                )
                return out.transpose(1, 2)
        except Exception:
            return q

    if sm_scale is None:
        sm_scale = D**-0.5

    try:
        q_int8, q_scale, _ = quantize_per_warp_int8(q, tensor_layout)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"\n--- ROCm CRASH IN QUANTIZE Q ---")
        traceback.print_exc()
        return q

    try:
        k_int8, k_scale, _ = quantize_per_block_int8(k, tensor_layout)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"\n--- ROCm CRASH IN QUANTIZE K ---")
        traceback.print_exc()
        return q

    try:
        output = run_fused_wmma(
            q_int8, k_int8, v, q_scale, k_scale, sm_scale, tensor_layout, use_inst_buf
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"\n--- ROCm CRASH IN WMMA EXECUTION ---")
        traceback.print_exc()
        return q

    del q_int8, q_scale, k_int8, k_scale
    return output
