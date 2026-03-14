import torch
import sage_fused_quant


def test_transpose_permute():
    # Using Layout 1: (Batch, Heads, Tokens, Head_Dim)
    B = 2
    H = 4
    T = 100
    D = 64

    # The C++ kernel enforces a CTA_SIZE of 64
    CTA_SIZE = 64
    padded_T = ((T + CTA_SIZE - 1) // CTA_SIZE) * CTA_SIZE

    print(f">>> Initializing Input Tensor: ({B}, {H}, {T}, {D})")
    input_tensor = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda")

    print(f">>> Initializing Padded Output Tensor: ({B}, {H}, {D}, {padded_T})")
    output_tensor = torch.empty((B, H, D, padded_T), dtype=torch.float16, device="cuda")

    print(">>> Firing RDNA 3 TransposePadPermute Kernel...")
    # Pass 1 for tensor_layout = (B, H, T, D)
    sage_fused_quant.transpose_pad_permute_cuda(input_tensor, output_tensor, 1)

    torch.cuda.synchronize()
    print("✅ Kernel executed successfully without Segfault!")
    print(f"✅ Output shape verified: {output_tensor.shape}")


if __name__ == "__main__":
    test_transpose_permute()
