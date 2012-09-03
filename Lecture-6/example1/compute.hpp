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
#elif defined(__MSVC)
	// msvc or msvc-compatible compiler
	#define CSE6230_RESTRICT __restrict
#else
	#warning Compiler is not recognized and restrict qualifier is not used.
	#define CSE6230_RESTRICT
#endif

typedef void (*vector_add_function)(const double*, const double*, double*, size_t);

extern "C" void vector_add_naive(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length);
extern "C" void vector_add_sse2(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length);
extern "C" void vector_add_sse2_load_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length);
extern "C" void vector_add_sse2_store_aligned(const double *CSE6230_RESTRICT xPointer, const double *CSE6230_RESTRICT yPointer, double *CSE6230_RESTRICT sumPointer, size_t length);
