import torch
import sage_fused_quant


def test_mean_scale_quant():
    B = 2
    H = 4
    T = 100
    D = 64

    print(f">>> Initializing FP16 Input Tensor: ({B}, {H}, {T}, {D})")
    input_tensor = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda")

    # 1. TEST: Scale Fuse Quant
    quant_out = torch.empty((B, H, T, D), dtype=torch.int8, device="cuda")
    # Fix: Must be float32 (torch::kFloat)
    scale_out = torch.empty((B, H, T), dtype=torch.float32, device="cuda")

    print("\n>>> Firing RDNA 3 Scale Fuse Quant Kernel...")
    try:
        sage_fused_quant.scale_fuse_quant_cuda(
            input_tensor, quant_out, scale_out, 1, 127.0, 1
        )
        torch.cuda.synchronize()
        print("✅ scale_fuse_quant_cuda executed successfully!")
    except Exception as e:
        print(f"❌ Failed scale_fuse_quant_cuda: {e}")

    # 2. TEST: Mean Scale Fuse Quant
    quant_out_mean = torch.empty((B, H, T, D), dtype=torch.int8, device="cuda")
    scale_out_mean = torch.empty((B, H, T), dtype=torch.float32, device="cuda")
    # Fix: The missing 4th tensor to catch the means
    mean_out = torch.empty((B, H, T), dtype=torch.float32, device="cuda")

    print("\n>>> Firing RDNA 3 Mean Scale Fuse Quant Kernel...")
    try:
        sage_fused_quant.mean_scale_fuse_quant_cuda(
            input_tensor, quant_out_mean, scale_out_mean, mean_out, 1, 127.0, 1
        )
        torch.cuda.synchronize()
        print("✅ mean_scale_fuse_quant_cuda executed successfully!")
    except Exception as e:
        print(f"❌ Failed mean_scale_fuse_quant_cuda: {e}")


if __name__ == "__main__":
    test_mean_scale_quant()
