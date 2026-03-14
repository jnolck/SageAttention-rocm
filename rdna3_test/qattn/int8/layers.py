import json
from safetensors import safe_open

source_file = "waiIllustriousSDXL_v160.safetensors"
layer_config = {}

print(f">>> Scanning {source_file}...")

with safe_open(source_file, framework="pt", device="cpu") as f:
    for key in f.keys():
        if key.startswith("model.diffusion_model."):
            # Target the UNet for INT8 SVD quantization
            layer_config[key] = "int8"
        else:
            # Force the VAE and Text Encoders to stay pristine
            layer_config[key] = (
                "skip"  # Depending on the tool, this might need to be "fp16" or "none"
            )

print(f">>> Mapped {len(layer_config)} total layers.")

with open("layers.json", "w") as f:
    json.dump(layer_config, f, indent=8)

print(">>> Saved layer targets to layers.json")
