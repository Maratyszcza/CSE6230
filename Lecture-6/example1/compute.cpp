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
#include <x86intrin.h>

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
