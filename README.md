Started with the official sage code. Ripped away any code that was not sm_80. Started with the headers and worked my way down to the kernels. Most of the heavy lifting was done by hipify-clang and Google's Gemmini. There's a lot of work to be done but it's numerically solid. Been running it for a couple of days without crashes. All the work was done on top of docker.io/rocm/pytorch. Most of the testing done has been with the qk_int8_sv_f16_accum_f32_attn and qk_int8_sv_f16_accum_f32_attn kernels. Little to no testing has been done with the qk_int8_sv_f16_accum_f32_attn kernel, and the qk_int8_sv_f16_accum_f32_attn kernel is still inactive. 

Max Absolute Error:  0.004395
Mean Absolute Error: 0.000313
