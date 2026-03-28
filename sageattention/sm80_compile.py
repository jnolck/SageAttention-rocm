import torch
from . import attn_sm80
from . import sage_fused_quant

# Route our python variables directly to the newly registered native ATen operations.
qk_int8_sv_f16_accum_f16_attn = torch.ops.attn_sm80.qk_int8_sv_f16_accum_f16_attn
qk_int8_sv_f16_accum_f32_attn = torch.ops.attn_sm80.qk_int8_sv_f16_accum_f32_attn
qk_int8_sv_f16_accum_f16_fuse_v_mean_attn = (
    torch.ops.attn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn
)

# QUANTIZATION WRAPPERS (EXACT NAMES)
quant_per_warp_int8_cuda = torch.ops.sage_fused_quant.quant_per_warp_int8_cuda
quant_per_block_int8_cuda = torch.ops.sage_fused_quant.quant_per_block_int8_cuda
quant_per_block_int8_fuse_sub_mean_cuda = (
    torch.ops.sage_fused_quant.quant_per_block_int8_fuse_sub_mean_cuda
)
sub_mean_cuda = torch.ops.sage_fused_quant.sub_mean_cuda
