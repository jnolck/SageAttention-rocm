from safetensors.torch import load_file

tensors = load_file("unet_heavy_learned_int8_tensorwise.safetensors")
for key in list(tensors.keys())[:10]:
    print(key)
