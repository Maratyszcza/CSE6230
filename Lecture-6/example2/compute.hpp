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

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
	// gcc or gcc-compatible compiler
	#define CSE6230_RESTRICT __restrict__
#elif defined(_MSC_VER)
	// msvc or msvc-compatible compiler
	#define CSE6230_RESTRICT __restrict
#else
	#warning Compiler is not recognized and restrict qualifier is not used.
	#define CSE6230_RESTRICT
#endif

#if defined(__GNUC__)
	#if defined(__SSE2__)
		#define CSE6230_SSE2_INTRINSICS_SUPPORTED
	#endif
	#if defined(__SSE3__)
		#define CSE6230_SSE3_INTRINSICS_SUPPORTED
	#endif
	#if defined(__AVX__)
		#define CSE6230_AVX_INTRINSICS_SUPPORTED
	#endif
	#if defined(__FMA4__)
		#define CSE6230_FMA4_INTRINSICS_SUPPORTED
	#endif
#elif defined(_MSC_VER)
	#if defined(_M_IX86) || defined(_M_X64)
		#define CSE6230_SSE2_INTRINSICS_SUPPORTED
		#define CSE6230_SSE3_INTRINSICS_SUPPORTED
		#define CSE6230_AVX_INTRINSICS_SUPPORTED
		#define CSE6230_FMA4_INTRINSICS_SUPPORTED
	#endif
#else
	#warning Compiler is not recognized and intrinsic functions are not used.
#endif


typedef void (*vector3d_dot_products_function)(const double*, const double*, double*, size_t);

extern "C" void vector3d_dot_products_naive(const double *CSE6230_RESTRICT v1Pointer, const double *CSE6230_RESTRICT v2Pointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount);
#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
extern "C" void vector3d_dot_products_sse2(const double *CSE6230_RESTRICT v1Pointer, const double *CSE6230_RESTRICT v2Pointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount);
#endif
#ifdef CSE6230_SSE3_INTRINSICS_SUPPORTED
extern "C" void vector3d_dot_products_sse3(const double *CSE6230_RESTRICT v1Pointer, const double *CSE6230_RESTRICT v2Pointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount);
#endif
#ifdef CSE6230_FMA4_INTRINSICS_SUPPORTED
extern "C" void vector3d_dot_products_fma4(const double *CSE6230_RESTRICT v1Pointer, const double *CSE6230_RESTRICT v2Pointer, double *CSE6230_RESTRICT dpPointer, size_t vectorsCount);
#endif
