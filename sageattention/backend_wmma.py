import torch
from . import attn_sm80


def run_fused_wmma(
    q_int8,
    k_int8,
    v,
    q_scale,
    k_scale,
    sm_scale,
    tensor_layout="NHD",
    use_inst_buf=False,
):
    v_c = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    v_c.copy_(v)

    output = torch.empty(v_c.shape, dtype=v_c.dtype, device=v_c.device)
    layout_int = 0 if tensor_layout == "NHD" else 1

    if use_inst_buf:
        pass
    else:
        attn_sm80.qk_int8_sv_f16_accum_f16_attn(
            q_int8,
            k_int8,
            v_c,
            output,
            q_scale,
            k_scale,
            layout_int,
            0,
            2,
            float(sm_scale),
            0,
        )

    return output
