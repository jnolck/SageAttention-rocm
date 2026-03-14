import torch
import time
from diffusers import StableDiffusionXLPipeline

# IMPORTING OUR BABY!
from sageattention.core import sageattn


# This is just the adapter so Diffusers lets us hijack the pipeline.
class SageAttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        b, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, b)
            attention_mask = attention_mask.view(
                b, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Format the matrices for our kernel
        head_dim = query.shape[-1] // attn.heads
        query = query.view(b, -1, attn.heads, head_dim)
        key = key.view(b, -1, attn.heads, head_dim)
        value = value.view(b, -1, attn.heads, head_dim)

        # ==========================================
        # 🔥 OUR BABY GIRL SAGE LIVES HERE 🔥
        # ==========================================
        hidden_states = sageattn(
            q=query,
            k=key,
            v=value,
            attn_mask=attention_mask,
            is_causal=False,
            tensor_layout="NHD",
        )
        # ==========================================

        hidden_states = hidden_states.reshape(b, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


print(">>> 1. Loading Pristine FP16 Monolithic Model...")
pipe = StableDiffusionXLPipeline.from_single_file(
    "./waiIllustriousSDXL_v160.safetensors",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

print(">>> 2. 🚀 INJECTING OUR SAGE ATTENTION KERNEL...")
# This forces Diffusers to abandon its math and use our SageAttnProcessor
pipe.unet.set_attn_processor(SageAttnProcessor())

prompt = "Aerith gainsborough, 1girl, cropped jacket, dress, drill hair, drill sidelocks, flower, forehead, hair ribbon, holding, holding flower, jacket, long hair, parted bangs, red jacket, ribbon, short sleeves, sidelocks, amano yoshitaka, abstract, watercolor ,masterpiece,best quality,amazing quality,"
steps = 20

print("\n>>> Warming up the AMD driver (5 steps)...")
_ = pipe(prompt, num_inference_steps=5)

print(f"\n>>> 🚀 STARTING {steps}-STEP SAGE-ATTENTION RUN...")
torch.cuda.synchronize()
start = time.time()

image = pipe(prompt, num_inference_steps=steps).images[0]

torch.cuda.synchronize()
end = time.time()

print(f"\n=======================================================")
print(f"🔥 SAGE ATTENTION FP16: {end - start:.2f} seconds 🔥")
print(f"🔥 SPEED: {steps / (end - start):.2f} it/s 🔥")
print(f"=======================================================\n")

image.save("aerith_sage_fp16.png")
print(">>> Image saved to aerith_sage_fp16.png")
