"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.hipify.hipify_python

# --- THE HOMER SIMPSON MANEUVER (Global) ---
# We couldn't find a proper way to shut this thing from trying run hipify
# on all the files.
torch.utils.hipify.hipify_python.hipify = lambda *args, **kwargs: {}
# -------------------------------------------

compiler_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
        "-w",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx1100",
        "-w",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
    ],
}
# "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1", "-fno-gpu-rdc",  # Keep linking simple

setup(
    name="sageattention",
    version="2.0.0+rocm",
    packages=find_packages(),
    ext_modules=[
        # 1. The Preprocessing Armory
        CUDAExtension(
            name="sageattention.sage_fused_quant",
            sources=[
                "csrc/fused/pybind.cpp",
                "csrc/fused/fused.hip",
            ],
            extra_compile_args=compiler_args,
        ),
        # 2. The Core Matrix Math Boss Room
        CUDAExtension(
            name="sageattention.attn_sm80",
            sources=[
                "csrc/qattn/pybind_sm80.cpp",
                "csrc/qattn/qk_int_sv_f16_cuda_sm80.hip",
                "csrc/qattn/attn_utils.hip",
                "csrc/qattn/attn_cuda_sm80.hip",
            ],
            extra_compile_args=compiler_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
