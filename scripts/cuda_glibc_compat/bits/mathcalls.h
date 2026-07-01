/* glibc >= 2.38 / CUDA 12.x: avoid cospi/sinpi/rsqrt noexcept redeclaration conflicts.
 * See: https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591
 */
#ifndef TRELLIS2_CUDA_GLIBC_MATHCALLS_H
#define TRELLIS2_CUDA_GLIBC_MATHCALLS_H

#pragma push_macro("cospi")
#pragma push_macro("sinpi")
#pragma push_macro("rsqrt")
#pragma push_macro("cospif")
#pragma push_macro("sinpif")
#pragma push_macro("rsqrtf")

#define cospi __compat_cospi_
#define sinpi __compat_sinpi_
#define rsqrt __compat_rsqrt_
#define cospif __compat_cospif_
#define sinpif __compat_sinpif_
#define rsqrtf __compat_rsqrtf_

#include_next <bits/mathcalls.h>

#undef __DECL_SIMD___compat_cospi_
#undef __DECL_SIMD___compat_cospi__f
#undef __DECL_SIMD___compat_cospi__l
#undef __DECL_SIMD___compat_cospi__f32
#undef __DECL_SIMD___compat_cospi__f64
#undef __DECL_SIMD___compat_cospi__f128
#undef __DECL_SIMD___compat_cospi__f32x
#undef __DECL_SIMD___compat_cospi__f64x
#undef __DECL_SIMD___compat_sinpi_
#undef __DECL_SIMD___compat_sinpi__f
#undef __DECL_SIMD___compat_sinpi__l
#undef __DECL_SIMD___compat_sinpi__f32
#undef __DECL_SIMD___compat_sinpi__f64
#undef __DECL_SIMD___compat_sinpi__f128
#undef __DECL_SIMD___compat_sinpi__f32x
#undef __DECL_SIMD___compat_sinpi__f64x
#undef __DECL_SIMD___compat_rsqrt_
#undef __DECL_SIMD___compat_rsqrt__f
#undef __DECL_SIMD___compat_rsqrt__l
#undef __DECL_SIMD___compat_rsqrt__f32
#undef __DECL_SIMD___compat_rsqrt__f64
#undef __DECL_SIMD___compat_rsqrt__f128
#undef __DECL_SIMD___compat_rsqrt__f32x
#undef __DECL_SIMD___compat_rsqrt__f64x
#undef __DECL_SIMD___compat_cospif_
#undef __DECL_SIMD___compat_cospif__f
#undef __DECL_SIMD___compat_cospif__l
#undef __DECL_SIMD___compat_cospif__f32
#undef __DECL_SIMD___compat_cospif__f64
#undef __DECL_SIMD___compat_cospif__f128
#undef __DECL_SIMD___compat_cospif__f32x
#undef __DECL_SIMD___compat_cospif__f64x
#undef __DECL_SIMD___compat_sinpif_
#undef __DECL_SIMD___compat_sinpif__f
#undef __DECL_SIMD___compat_sinpif__l
#undef __DECL_SIMD___compat_sinpif__f32
#undef __DECL_SIMD___compat_sinpif__f64
#undef __DECL_SIMD___compat_sinpif__f128
#undef __DECL_SIMD___compat_sinpif__f32x
#undef __DECL_SIMD___compat_sinpif__f64x
#undef __DECL_SIMD___compat_rsqrtf_
#undef __DECL_SIMD___compat_rsqrtf__f
#undef __DECL_SIMD___compat_rsqrtf__l
#undef __DECL_SIMD___compat_rsqrtf__f32
#undef __DECL_SIMD___compat_rsqrtf__f64
#undef __DECL_SIMD___compat_rsqrtf__f128
#undef __DECL_SIMD___compat_rsqrtf__f32x
#undef __DECL_SIMD___compat_rsqrtf__f64x

#pragma pop_macro("rsqrtf")
#pragma pop_macro("sinpif")
#pragma pop_macro("cospif")
#pragma pop_macro("rsqrt")
#pragma pop_macro("sinpi")
#pragma pop_macro("cospi")

#endif
