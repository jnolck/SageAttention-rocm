from safetensors.torch import load_file, save_file

print(">>> Loading original model...")
state_dict = load_file("plantMilkModelSuite_walnut.safetensors")

unet_heavy = {}
host_safe = {}

for k, v in state_dict.items():
    # Only grab the heavy UNet layers
    if k.startswith("model.diffusion_model.") and not (
        k.endswith(".bias") or "norm" in k or "time_embed" in k or "label_emb" in k
    ):
        unet_heavy[k] = v
    else:
        # Put the VAE, Text Encoders, and sensitive 1D arrays in the vault
        host_safe[k] = v

save_file(unet_heavy, "unet_heavy.safetensors")
save_file(host_safe, "host_safe.safetensors")
print(">>> Split complete. Safe components locked in host_safe.safetensors.")
