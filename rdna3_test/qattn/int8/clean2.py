import json

print(">>> Loading layers.json...")
with open("layers.json", "r") as f:
    data = json.load(f)

cleaned_data = {}
for key, value in data.items():
    fmt = value.get("format", value) if isinstance(value, dict) else value

    # 1. Drop the explicit skips
    if fmt == "skip":
        continue

    # 2. THE FIX: Ruthlessly drop all 1D biases, LayerNorms, and time embeddings.
    # These crash SVD and save zero VRAM.
    if (
        key.endswith(".bias")
        or "norm" in key
        or "time_embed" in key
        or "label_emb" in key
    ):
        continue

    # 3. Fix the INT8 naming convention
    if fmt == "int8":
        fmt = "int8_blockwise"

    cleaned_data[key] = {"format": fmt}

print(f">>> Kept {len(cleaned_data)} heavy UNet matrices for INT8 quantization.")

with open("layers.json", "w") as f:
    json.dump(cleaned_data, f, indent=8)

print(">>> layers.json is clean, safely filtered, and ready.")
