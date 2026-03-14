import torch
import sage_fused_quant


def test_hardware_quantization():
    print(">>> INITIALIZING RDNA 3 QUANTIZATION QUARANTINE <<<")

    # 1. Setup mock dimensions (mimicking a 512x512 SD1.5 cross-attention block)
    B, H, S, D = 2, 10, 1024, 64
    BLOCK_SIZE = 128
    LAYOUT = 1  # 1 = [Batch, Heads, Seq, Dim]

    # 2. Generate fake FP16 neural network data
    print(f"Generating fake FP16 tensor: [{B}, {H}, {S}, {D}]")
    input_fp16 = torch.randn(
        (B, H, S, D), dtype=torch.float16, device="cuda"
    ).contiguous()

    # 3. Pre-allocate the empty memory for the C++ kernel
    # Scale shape must be [B, H, (S + BLOCK_SIZE - 1) // BLOCK_SIZE]
    scale_len = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    output_int8_hw = torch.empty_like(input_fp16, dtype=torch.int8).contiguous()
    scale_fp32_hw = torch.empty(
        (B, H, scale_len), dtype=torch.float32, device="cuda"
    ).contiguous()

    # 4. Fire the bare-metal kernel
    print("Dispatching to RDNA 3 hardware...")
    try:
        # Using the 5-argument overload: (input, output, scale, block_size, layout)
        sage_fused_quant.quant_per_block_int8_cuda(
            input_fp16, output_int8_hw, scale_fp32_hw, BLOCK_SIZE, LAYOUT
        )
        print("✅ Hardware execution completed without segfaulting!")
    except Exception as e:
        print(f"❌ HARDWARE CRASH: {e}")
        return

    # 5. The Verification (Python vs Hardware)
    print("Verifying hardware math against Python baseline...")

    # We test the first block of the first head
    test_block_fp16 = input_fp16[0, 0, 0:BLOCK_SIZE, :]

    # Our slow Python math from yesterday
    py_max = test_block_fp16.abs().max().clamp(min=1e-5).to(torch.float32)
    py_scale = py_max / 127.0
    py_int8 = torch.round(test_block_fp16 / py_scale).to(torch.int8)

    # The Hardware's results
    hw_scale = scale_fp32_hw[0, 0, 0]
    hw_int8 = output_int8_hw[0, 0, 0:BLOCK_SIZE, :]

    print(f"Python Scale  : {py_scale.item():.6f}")
    print(f"Hardware Scale: {hw_scale.item():.6f}")

    # Check if the hardware matched Python (allowing for tiny floating point rounding differences)
    if torch.allclose(py_scale, hw_scale, atol=1e-4):
        print("✅ Scale calculation is MATHEMATICALLY PERFECT.")
    else:
        print("❌ Scale mismatch!")


if __name__ == "__main__":
    test_hardware_quantization()
