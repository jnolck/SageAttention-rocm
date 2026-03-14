import json

print(">>> Loading layers.json...")
with open("layers.json", "r") as f:
    data = json.load(f)

cleaned_data = {}
for key, value in data.items():
    # Extract the string whether it is raw or inside a dict
    fmt = value.get("format", value) if isinstance(value, dict) else value

    # 1. Ruthlessly drop the skips
    if fmt == "skip":
        continue

    # 2. Fix the INT8 naming convention to match the strict validator
    if fmt == "int8":
        fmt = "int8_blockwise"

    cleaned_data[key] = {"format": fmt}

print(f">>> Kept {len(cleaned_data)} UNet layers for INT8 quantization.")

with open("layers.json", "w") as f:
    json.dump(cleaned_data, f, indent=8)

print(">>> layers.json is clean, valid, and ready.")
