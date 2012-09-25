/******************************************************************************\
 *                                                                            *
 * Copyright (c) 2012 Marat Dukhan                                            *
 *                                                                            *
 * This software is provided 'as-is', without any express or implied          *
 * warranty. In no event will the authors be held liable for any damages      *
 * arising from the use of this software.                                     *
 *                                                                            *
 * Permission is granted to anyone to use this software for any purpose,      *
 * including commercial applications, and to alter it and redistribute it     *
 * freely, subject to the following restrictions:                             *
 *                                                                            *
 * 1. The origin of this software must not be misrepresented; you must not    *
 * claim that you wrote the original software. If you use this software       *
 * in a product, an acknowledgment in the product documentation would be      *
 * appreciated but is not required.                                           *
 *                                                                            *
 * 2. Altered source versions must be plainly marked as such, and must not be *
 * misrepresented as being the original software.                             *
 *                                                                            *
 * 3. This notice may not be removed or altered from any source               *
 * distribution.                                                              *
 *                                                                            *
\******************************************************************************/

#include <compute.hpp>
#include <math.h>
#if defined(CSE6230_SSE2_INTRINSICS_SUPPORTED) || defined(CSE6230_AVX_INTRINSICS_SUPPORTED)
	#if defined(__GNUC__)
		#include <x86intrin.h>
	#elif defined(_MSC_VER)
		#include <intrin.h>
	#else
		#error Intrinsics headers are not included: unknown compiler
	#endif
#endif

void vector3d_dot_products_naive(const double *CSE6230_RESTRICT vPointer, const double *CSE6230_RESTRICT uPointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount) {
	for (; vectorsCount != 0; vectorsCount -= 1) {
		const double vX = vPointer[0];
		const double vY = vPointer[1];
		const double vZ = vPointer[2];
		
		const double uX = uPointer[0];
		const double uY = uPointer[1];
		const double uZ = uPointer[2];

		const double dotProduct = vX * uX + vY * uY + vZ * uZ;
		*dpPointer = dotProduct;
		
		// Advance pointers to the next 3-element vectors
		vPointer += 3;
		uPointer += 3;
		// Advance pointer to the next dot product
		dpPointer += 1;
	}
}

#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
void vector3d_dot_products_sse2(const double *CSE6230_RESTRICT vPointer, const double *CSE6230_RESTRICT uPointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount) {
	// Process arrays by two elements at an iteration
	for (; vectorsCount >= 2; vectorsCount -= 2) {
		// Load two V vectors
		const __m128d v0X_v0Y = _mm_loadu_pd(vPointer);
		const __m128d v0Z_v1X = _mm_loadu_pd(vPointer + 2);
		const __m128d v1Y_v1Z = _mm_loadu_pd(vPointer + 4);
		
		// Load two U vectors
		const __m128d u0X_u0Y = _mm_loadu_pd(uPointer);
		const __m128d u0Z_u1X = _mm_loadu_pd(uPointer + 2);
		const __m128d u1Y_u1Z = _mm_loadu_pd(uPointer + 4);
		
		// Multiply corresponding coordinates
		const __m128d uv0X_uv0Y = _mm_mul_pd(v0X_v0Y, u0X_u0Y);
		const __m128d uv0Z_uv1X = _mm_mul_pd(v0Z_v1X, u0Z_u1X);
		const __m128d uv1Y_uv1Z = _mm_mul_pd(v1Y_v1Z, u1Y_u1Z);
		
		const __m128d uv0X_uv1Y = _mm_unpacklo_pd(uv0Z_uv1X, uv1Y_uv1Z);
		const __m128d uv0Y_uv1Z = _mm_unpackhi_pd(uv0X_uv0Y, uv1Y_uv1Z);
		
		const __m128d dp0_dp1 = _mm_add_pd(_mm_add_pd(uv0X_uv1Y, uv0Y_uv1Z), uv0Z_uv1X);
		
		_mm_storeu_pd(dpPointer, dp0_dp1); // Store two dot products
		
		// Advance pointers to the next two elements
		vPointer += 6;
		uPointer += 6;
		dpPointer += 2;
	}
	// Process remaining vectors (if any)
	for (; vectorsCount != 0; vectorsCount -= 1) {
		const double vX = vPointer[0];
		const double vY = vPointer[1];
		const double vZ = vPointer[2];
		
		const double uX = uPointer[0];
		const double uY = uPointer[1];
		const double uZ = uPointer[2];

		const double dotProduct = vX * uX + vY * uY + vZ * uZ;
		*dpPointer = dotProduct;
		
		// Advance pointers to the next 3-element vectors
		vPointer += 3;
		uPointer += 3;
		// Advance pointer to the next dot product
		dpPointer += 1;
	}
}
#endif

#ifdef CSE6230_SSE3_INTRINSICS_SUPPORTED
void vector3d_dot_products_sse3(const double *CSE6230_RESTRICT vPointer, const double *CSE6230_RESTRICT uPointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount) {
	// Process arrays by two elements at an iteration
	for (; vectorsCount >= 2; vectorsCount -= 2) {
		// Load two V vectors
		const __m128d v0X_v0Y = _mm_loadu_pd(vPointer);
		const __m128d v0Z_v1X = _mm_loadu_pd(vPointer + 2);
		const __m128d v1Y_v1Z = _mm_loadu_pd(vPointer + 4);
		
		// Load two U vectors
		const __m128d u0X_u0Y = _mm_loadu_pd(uPointer);
		const __m128d u0Z_u1X = _mm_loadu_pd(uPointer + 2);
		const __m128d u1Y_u1Z = _mm_loadu_pd(uPointer + 4);
		
		// Multiply corresponding coordinates
		const __m128d uv0X_uv0Y = _mm_mul_pd(v0X_v0Y, u0X_u0Y);
		const __m128d uv0Z_uv1X = _mm_mul_pd(v0Z_v1X, u0Z_u1X);
		const __m128d uv1Y_uv1Z = _mm_mul_pd(v1Y_v1Z, u1Y_u1Z);
		
		const __m128d dp0_dp1 = _mm_add_pd(_mm_hadd_pd(uv0X_uv0Y, uv1Y_uv1Z), uv0Z_uv1X);

		_mm_storeu_pd(dpPointer, dp0_dp1); // Store two dot products
		
		// Advance pointers to the next two elements
		vPointer += 6;
		uPointer += 6;
		dpPointer += 2;
	}
	// Process remaining vectors (if any)
	for (; vectorsCount != 0; vectorsCount -= 1) {
		const double vX = vPointer[0];
		const double vY = vPointer[1];
		const double vZ = vPointer[2];
		
		const double uX = uPointer[0];
		const double uY = uPointer[1];
		const double uZ = uPointer[2];

		const double dotProduct = vX * uX + vY * uY + vZ * uZ;
		*dpPointer = dotProduct;
		
		// Advance pointers to the next 3-element vectors
		vPointer += 3;
		uPointer += 3;
		// Advance pointer to the next dot product
		dpPointer += 1;
	}
}
#endif

#ifdef CSE6230_FMA4_INTRINSICS_SUPPORTED
void vector3d_dot_products_fma4(const double *CSE6230_RESTRICT vPointer, const double *CSE6230_RESTRICT uPointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount) {
	// Process arrays by two elements at an iteration
	for (; vectorsCount >= 2; vectorsCount -= 2) {
		// Load two V vectors
		const __m128d v0X_v0Y = _mm_loadu_pd(vPointer);
		const __m128d v0Z_v1X = _mm_loadu_pd(vPointer + 2);
		const __m128d v1Y_v1Z = _mm_loadu_pd(vPointer + 4);
		
		// Load two U vectors
		const __m128d u0X_u0Y = _mm_loadu_pd(uPointer);
		const __m128d u0Z_u1X = _mm_loadu_pd(uPointer + 2);
		const __m128d u1Y_u1Z = _mm_loadu_pd(uPointer + 4);
		
		// Shuffle vectors
		const __m128d v0X_v1Y = _mm_unpacklo_pd(v0X_v0Y, v1Y_v1Z);
		const __m128d u0X_u1Y = _mm_unpacklo_pd(u0X_u0Y, u1Y_u1Z);
		const __m128d v0Y_v1Z = _mm_unpackhi_pd(v0X_v0Y, v1Y_v1Z);
		const __m128d u0Y_u1Z = _mm_unpackhi_pd(u0X_u0Y, u1Y_u1Z);
		
		// Multiply corresponding coordinates
		const __m128d uv0Z_uv1X = _mm_mul_pd(v0Z_v1X, u0Z_u1X);
		
		// Multiply-accumulate full dot products
		const __m128d dp0_dp1 = _mm_macc_pd(v0Y_v1Z, u0Y_u1Z, _mm_macc_pd(v0X_v1Y, u0X_u1Y, uv0Z_uv1X));
		
		_mm_storeu_pd(dpPointer, dp0_dp1); // Store two dot products
		
		// Advance pointers to the next two elements
		vPointer += 6;
		uPointer += 6;
		dpPointer += 2;
	}
	// Process remaining vectors (if any)
	for (; vectorsCount != 0; vectorsCount -= 1) {
		const double vX = vPointer[0];
		const double vY = vPointer[1];
		const double vZ = vPointer[2];
		
		const double uX = uPointer[0];
		const double uY = uPointer[1];
		const double uZ = uPointer[2];

		const double dotProduct = vX * uX + vY * uY + vZ * uZ;
		*dpPointer = dotProduct;
		
		// Advance pointers to the next 3-element vectors
		vPointer += 3;
		uPointer += 3;
		// Advance pointer to the next dot product
		dpPointer += 1;
	}
}
#endif

