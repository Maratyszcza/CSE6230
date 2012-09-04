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

inline static double minus_inf() {
	#if defined(__GNUC__)
		return -__builtin_inf();
	#else
		static const double plus_zero = +0.0;
		static const double minus_one = -1.0;
		return plus_zero / minus_one;
	#endif
}

void vector_add_naive(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}

#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
void vector_add_sse2(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process arrays by two elements at an iteration
	for (; length >= 2; length -= 2) {
		const __m128d x = _mm_loadu_pd(xPointer); // Load two x elements
		const __m128d y = _mm_loadu_pd(yPointer); // Load two y elements
		const __m128d sum = _mm_add_pd(x, y); // Compute two sum elements
		_mm_storeu_pd(sumPointer, sum); // Store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 2;
		yPointer += 2;
		sumPointer += 2;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
void vector_add_sse2_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process arrays by two elements at an iteration
	for (; length >= 2; length -= 2) {
		const __m128d x = _mm_load_pd(xPointer); // Aligned (!) load two x elements
		const __m128d y = _mm_load_pd(yPointer); // Aligned (!) load two y elements
		const __m128d sum = _mm_add_pd(x, y); // Compute two sum elements
		_mm_store_pd(sumPointer, sum); // Aligned (!) store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 2;
		yPointer += 2;
		sumPointer += 2;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
void vector_add_sse2_load_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process by one element until xPointer (the first input array) is aligned on 16
	for (; (size_t(xPointer) % size_t(16) != 0) && (length != 0); length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
	// Process arrays by two elements at an iteration
	// xPointer is aligned on 16, so we can use aligned load instruction
	for (; length >= 2; length -= 2) {
		const __m128d x = _mm_load_pd(xPointer); // Aligned (!) load two x elements
		const __m128d y = _mm_loadu_pd(yPointer); // Load two y elements
		const __m128d sum = _mm_add_pd(x, y); // Compute two sum elements
		_mm_storeu_pd(sumPointer, sum); // Store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 2;
		yPointer += 2;
		sumPointer += 2;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
void vector_add_sse2_store_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process by one element until sumPointer (the output array) is aligned on 16
	for (; (size_t(sumPointer) % size_t(16) != 0) && (length != 0); length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
	// Process arrays by two elements at an iteration
	// sumPointer is aligned on 16, so we can use aligned store instruction
	for (; length >= 2; length -= 2) {
		const __m128d x = _mm_loadu_pd(xPointer); // Load two x elements
		const __m128d y = _mm_loadu_pd(yPointer); // Load two y elements
		const __m128d sum = _mm_add_pd(x, y); // Compute two sum elements
		_mm_store_pd(sumPointer, sum); // Aligned (!) store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 2;
		yPointer += 2;
		sumPointer += 2;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
void vector_add_avx(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process arrays by two elements at an iteration
	for (; length >= 4; length -= 4) {
		const __m256d x = _mm256_loadu_pd(xPointer); // Load two x elements
		const __m256d y = _mm256_loadu_pd(yPointer); // Load two y elements
		const __m256d sum = _mm256_add_pd(x, y); // Compute two sum elements
		_mm256_storeu_pd(sumPointer, sum); // Store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 4;
		yPointer += 4;
		sumPointer += 4;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
void vector_add_avx_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process arrays by two elements at an iteration
	for (; length >= 4; length -= 4) {
		const __m256d x = _mm256_load_pd(xPointer); // Aligned (!) load two x elements
		const __m256d y = _mm256_load_pd(yPointer); // Aligned (!) load two y elements
		const __m256d sum = _mm256_add_pd(x, y); // Compute two sum elements
		_mm256_store_pd(sumPointer, sum); // Aligned (!) store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 4;
		yPointer += 4;
		sumPointer += 4;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
void vector_add_avx_load_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process by one element until xPointer (the first input array) is aligned on 32
	for (; (size_t(xPointer) % size_t(32) != 0) && (length != 0); length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
	// Process arrays by two elements at an iteration
	// xPointer is aligned on 32, so we can use aligned load instruction
	for (; length >= 4; length -= 4) {
		const __m256d x = _mm256_load_pd(xPointer); // Aligned (!) load two x elements
		const __m256d y = _mm256_loadu_pd(yPointer); // Load two y elements
		const __m256d sum = _mm256_add_pd(x, y); // Compute two sum elements
		_mm256_storeu_pd(sumPointer, sum); // Store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 4;
		yPointer += 4;
		sumPointer += 4;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
void vector_add_avx_store_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length) {
	// Process by one element until sumPointer (the output array) is aligned on 32
	for (; (size_t(sumPointer) % size_t(32) != 0) && (length != 0); length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
	// Process arrays by two elements at an iteration
	// sumPointer is aligned on 16, so we can use aligned store instruction
	for (; length >= 4; length -= 4) {
		const __m256d x = _mm256_loadu_pd(xPointer); // Load two x elements
		const __m256d y = _mm256_loadu_pd(yPointer); // Load two y elements
		const __m256d sum = _mm256_add_pd(x, y); // Compute two sum elements
		_mm256_store_pd(sumPointer, sum); // Aligned (!) store two sum elements
		
		// Advance pointers to the next two elements
		xPointer += 4;
		yPointer += 4;
		sumPointer += 4;
	}
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double x = *xPointer; // Load x
		const double y = *yPointer; // Load y
		const double sum = x + y; // Compute sum
		*sumPointer = sum; // Store sum

		// Advance pointers to the next elements
		xPointer += 1;
		yPointer += 1;
		sumPointer += 1;
	}
}
#endif

void vector_max_naive(const double *CSE6230_RESTRICT arrayPointer, double *CSE6230_RESTRICT maxPointer, size_t length) {
	double max = minus_inf();
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double element = *arrayPointer; // Load array elements
		max = fmax(max, element);

		// Advance pointers to the next element
		arrayPointer += 1;
	}
	*maxPointer = max;
}

#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
void vector_max_sse2(const double *CSE6230_RESTRICT arrayPointer, double *CSE6230_RESTRICT maxPointer, size_t length) {
	// Process arrays by two elements at an iteration
	__m128d maxX2 = _mm_set1_pd(minus_inf());
	for (; length >= 2; length -= 2) {
		const __m128d elementX2 = _mm_loadu_pd(arrayPointer); // Load two array elements
		maxX2 = _mm_max_pd(maxX2, elementX2);
		
		// Advance pointers to the next two elements
		arrayPointer += 2;
	}
	const __m128d maxX2High = _mm_unpackhi_pd(maxX2, maxX2); // Both elements of maxX2High contain the high part of maxX2
	const __m128d maxX2Reduced = _mm_max_sd(maxX2, maxX2High); // The low element of maxX2Reduced contains the max of two elements in maxX2
	double max = _mm_cvtsd_f64(maxX2Reduced);
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double element = *arrayPointer; // Load array elements
		max = fmax(max, element);

		// Advance pointers to the next element
		arrayPointer += 1;
	}
	*maxPointer = max;
}
#endif

#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
void vector_max_sse2_load_aligned(const double *CSE6230_RESTRICT arrayPointer, double *CSE6230_RESTRICT maxPointer, size_t length) {
	double max = minus_inf();
	// Process by one element until arrayPointer (the input array) is aligned on 16
	for (; (size_t(arrayPointer) % size_t(16) != 0) && (length != 0); length -= 1) {
		const double element = *arrayPointer; // Load array element
		max = fmax(max, element);

		// Advance pointer to the next elements
		arrayPointer += 1;
	}
	// Process arrays by two elements at an iteration
	__m128d maxX2 = _mm_set1_pd(max);
	for (; length >= 2; length -= 2) {
		const __m128d elementX2 = _mm_load_pd(arrayPointer); // Aligned (!) load two array elements
		maxX2 = _mm_max_pd(maxX2, elementX2);
		
		// Advance pointers to the next two elements
		arrayPointer += 2;
	}
	const __m128d maxX2High = _mm_unpackhi_pd(maxX2, maxX2); // Both elements of maxX2High contain the high part of maxX2
	const __m128d maxX2Reduced = _mm_max_sd(maxX2, maxX2High); // The low element of maxX2Reduced contains the max of two elements in maxX2
	max = _mm_cvtsd_f64(maxX2Reduced);
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double element = *arrayPointer; // Load array element
		max = fmax(max, element);

		// Advance pointers to the next element
		arrayPointer += 1;
	}
	*maxPointer = max;
}
#endif

#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
void vector_max_avx(const double *CSE6230_RESTRICT arrayPointer, double *CSE6230_RESTRICT maxPointer, size_t length) {
	// Process arrays by four elements at an iteration
	__m256d maxX4 = _mm256_set1_pd(minus_inf());
	for (; length >= 4; length -= 4) {
		const __m256d elementX4 = _mm256_loadu_pd(arrayPointer); // Load four array elements
		maxX4 = _mm256_max_pd(maxX4, elementX4);
		
		// Advance pointers to the next four elements
		arrayPointer += 4;
	}
	const __m128d maxX4High = _mm256_extractf128_pd(maxX4, 1); // Contains the two high elements of maxX4
	const __m128d maxX4Low = _mm256_castpd256_pd128(maxX4); // Contains the two low elements of maxX4
	const __m128d maxX4PartiallyReduced = _mm_max_pd(maxX4Low, maxX4High); // Contains two elements - partially reduced sum of four elements in maxX4
	const __m128d maxX4PartiallyReducedHigh = _mm_unpackhi_pd(maxX4PartiallyReduced, maxX4PartiallyReduced); // Both elements of maxX4PartiallyReducedHigh contain the high part of maxX4PartiallyReduced
	const __m128d maxX4Reduced = _mm_max_sd(maxX4PartiallyReduced, maxX4PartiallyReducedHigh); // The low element of maxX4Reduced contains the max of four elements in maxX4
	double max = _mm_cvtsd_f64(maxX4Reduced);
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double element = *arrayPointer; // Load array elements
		max = fmax(max, element);

		// Advance pointers to the next element
		arrayPointer += 1;
	}
	*maxPointer = max;
}
#endif

#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
void vector_max_avx_load_aligned(const double *CSE6230_RESTRICT arrayPointer, double *CSE6230_RESTRICT maxPointer, size_t length) {
	double max = minus_inf();
	// Process by one element until arrayPointer (the input array) is aligned on 32
	for (; (size_t(arrayPointer) % size_t(32) != 0) && (length != 0); length -= 1) {
		const double element = *arrayPointer; // Load array element
		max = fmax(max, element);

		// Advance pointer to the next elements
		arrayPointer += 1;
	}
	// Process arrays by four elements at an iteration
	__m256d maxX4 = _mm256_set1_pd(max);
	for (; length >= 4; length -= 4) {
		const __m256d elementX4 = _mm256_load_pd(arrayPointer); // Aligned (!) load four array elements
		maxX4 = _mm256_max_pd(maxX4, elementX4);
		
		// Advance pointers to the next four elements
		arrayPointer += 4;
	}
	const __m128d maxX4High = _mm256_extractf128_pd(maxX4, 1); // Contains the two high elements of maxX4
	const __m128d maxX4Low = _mm256_castpd256_pd128(maxX4); // Contains the two low elements of maxX4
	const __m128d maxX4PartiallyReduced = _mm_max_pd(maxX4Low, maxX4High); // Contains two elements - partially reduced sum of four elements in maxX4
	const __m128d maxX4PartiallyReducedHigh = _mm_unpackhi_pd(maxX4PartiallyReduced, maxX4PartiallyReduced); // Both elements of maxX4PartiallyReducedHigh contain the high part of maxX4PartiallyReduced
	const __m128d maxX4Reduced = _mm_max_sd(maxX4PartiallyReduced, maxX4PartiallyReducedHigh); // The low element of maxX4Reduced contains the max of four elements in maxX4
	max = _mm_cvtsd_f64(maxX4Reduced);
	// Process remaining elements (if any)
	for (; length != 0; length -= 1) {
		const double element = *arrayPointer; // Load array elements
		max = fmax(max, element);

		// Advance pointers to the next element
		arrayPointer += 1;
	}
	*maxPointer = max;
}
#endif
