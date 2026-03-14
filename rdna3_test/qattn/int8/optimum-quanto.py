import torch
from diffusers import StableDiffusionXLPipeline
from optimum.quanto import quantize, freeze, qint8

print(">>> Loading pristine FP16 model...")
pipe = StableDiffusionXLPipeline.from_single_file(
    "plantMilkModelSuite_walnut.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

print(">>> Quantizing the UNet to Native INT8...")
# This safely targets only the heavy linear/conv layers in the UNet
quantize(pipe.unet, weights=qint8)

print(">>> Freezing weights and scales...")
# Freeze locks the integer weights and float scales into permanent tensors
freeze(pipe.unet)

print(">>> Saving Hugging Face compatible INT8 UNet...")
pipe.unet.save_pretrained("walnut_unet_quanto_int8")
