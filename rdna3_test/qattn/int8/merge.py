from safetensors.torch import load_file, save_file

print(">>> Loading safe components...")
final_dict = load_file("host_safe.safetensors")

print(">>> Loading quantized UNet...")
# Hooking up the exact file the tool just generated
quant_dict = load_file("unet_heavy_learned_int8_tensorwise.safetensors")

print(">>> Fusing the components...")
final_dict.update(quant_dict)

output_name = "walnut_int8_final.safetensors"
save_file(final_dict, output_name)

print(f">>> Fused! Final perfectly mixed-precision model saved to: {output_name}")
