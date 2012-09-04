#include <compute.hpp>
#include <stdio.h>
#include <malloc.h>

inline static uint64_t get_cpu_ticks_start() {
#ifdef __x86_64__
	uint32_t low, high;
	__asm__ __volatile__ (
		"xor %%eax, %%eax;"
		"cpuid;"
		"rdtsc;"
	: "=a"(low), "=d"(high)
	:
	: "%rbx", "%rcx"
	);
	return (uint64_t(high) << 32) | uint64_t(low);
#endif
}

inline static uint64_t get_cpu_ticks_end() {
	return get_cpu_ticks_start();
}

inline static uint64_t min(uint64_t a, uint64_t b) {
	return a < b ? a : b;
}

inline static uint64_t max(uint64_t a, uint64_t b) {
	return a > b ? a : b;
}

static uint64_t time_vector_add(vector_add_function vector_add, const double* x_array, const double* y_array, double* sum_array, size_t array_size, size_t experiments_count) {
	uint64_t best_ticks = uint64_t(-1);
	for (size_t experiment_number = 1; experiment_number <= experiments_count; experiment_number++) {
		const uint64_t start_ticks = get_cpu_ticks_start();
		vector_add(x_array, y_array, sum_array, array_size);
		const uint64_t end_ticks = get_cpu_ticks_end();
		const uint64_t elapsed_ticks = end_ticks - start_ticks;
		best_ticks = min(best_ticks, elapsed_ticks);
	}
	return best_ticks;
}

static uint64_t time_vector_max(vector_max_function vector_max, const double* elements_array, size_t array_size, size_t experiments_count) {
	uint64_t best_ticks = uint64_t(-1);
	for (size_t experiment_number = 1; experiment_number <= experiments_count; experiment_number++) {
		double max_element;
		const uint64_t start_ticks = get_cpu_ticks_start();
		vector_max(elements_array, &max_element, array_size);
		const uint64_t end_ticks = get_cpu_ticks_end();
		const uint64_t elapsed_ticks = end_ticks - start_ticks;
		best_ticks = min(best_ticks, elapsed_ticks);
	}
	return best_ticks;
}

static void report_timings(const char* method_name, uint64_t aligned_ticks, uint64_t min_ticks, uint64_t max_ticks, size_t array_size) {
	printf("%20s\t%2.2lf\t%2.2lf\t%2.2lf\n", method_name,
		double(aligned_ticks) / double(array_size),
		double(min_ticks) / double(array_size),
		double(max_ticks) / double(array_size)
	);
}

static void test_vector_add(const char* method_name, vector_add_function vector_add, const double* x_array, const double* y_array, double* sum_array, size_t array_size, size_t experiments_count, size_t misalignment_bound) {
	const uint64_t aligned_vector_add_ticks = time_vector_add(vector_add, x_array, y_array, sum_array, array_size, experiments_count);
	uint64_t min_vector_add_ticks = uint64_t(-1);
	uint64_t max_vector_add_ticks = 0;
	for (size_t x_array_misalignment = 0; x_array_misalignment < misalignment_bound / sizeof(double); x_array_misalignment += 1) {
		for (size_t y_array_misalignment = 0; y_array_misalignment < misalignment_bound / sizeof(double); y_array_misalignment += 1) {
			for (size_t sum_array_misalignment = 0; sum_array_misalignment < misalignment_bound / sizeof(double); sum_array_misalignment += 1) {
				const uint64_t vector_add_ticks = time_vector_add(vector_add,
					x_array + x_array_misalignment,
					y_array + y_array_misalignment,
					sum_array + sum_array_misalignment,
					array_size, experiments_count);
				min_vector_add_ticks = min(min_vector_add_ticks, vector_add_ticks);
				max_vector_add_ticks = max(max_vector_add_ticks, vector_add_ticks);
			}
		}
	}
	report_timings(method_name, aligned_vector_add_ticks, min_vector_add_ticks, max_vector_add_ticks, array_size);
}

static void test_vector_max(const char* method_name, vector_max_function vector_max, const double* x_array, size_t array_size, size_t experiments_count, size_t misalignment_bound) {
	const uint64_t aligned_vector_max_ticks = time_vector_max(vector_max, x_array, array_size, experiments_count);
	uint64_t min_vector_max_ticks = uint64_t(-1);
	uint64_t max_vector_max_ticks = 0;
	for (size_t x_array_misalignment = 0; x_array_misalignment < misalignment_bound / sizeof(double); x_array_misalignment += 1) {
		const uint64_t vector_max_ticks = time_vector_max(vector_max,
			x_array + x_array_misalignment,
			array_size, experiments_count);
		min_vector_max_ticks = min(min_vector_max_ticks, vector_max_ticks);
		max_vector_max_ticks = max(max_vector_max_ticks, vector_max_ticks);
	}
	report_timings(method_name, aligned_vector_max_ticks, min_vector_max_ticks, max_vector_max_ticks, array_size);
}

static void report_timings(const char* method_name, uint64_t aligned_ticks, size_t array_size) {
	printf("%20s\t%2.2lf\n", method_name, double(aligned_ticks) / double(array_size));
}

int main(int argc, char** argv) {
	size_t experiments_count = 10000000;
	
	size_t array_size = 500;
	double *x_array = (double*)memalign(32, array_size * sizeof(double) + 32);
	double *y_array = (double*)memalign(32, array_size * sizeof(double) + 32);
	double *sum_array = (double*)memalign(32, array_size * sizeof(double) + 32);
	
	printf("Add Method\tAligned CPE\tMin CPE\tMax CPE\n");
	
	test_vector_add("Naive", &vector_add_naive, x_array, y_array, sum_array, array_size, experiments_count, 16);
	
	#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
	test_vector_add("SSE2", &vector_add_sse2, x_array, y_array, sum_array, array_size, experiments_count, 16);
	#endif
	
	#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
	const uint64_t aligned_vector_add_sse2_aligned_ticks = time_vector_add(&vector_add_sse2_aligned, x_array, y_array, sum_array, array_size, experiments_count);
	report_timings("SSE2 + aligned array", aligned_vector_add_sse2_aligned_ticks, array_size);
	#endif
	
	#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
	test_vector_add("SSE2 + aligned load", &vector_add_sse2_load_aligned, x_array, y_array, sum_array, array_size, experiments_count, 16);
	#endif
	
	#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
	test_vector_add("SSE2 + aligned store", &vector_add_sse2_store_aligned, x_array, y_array, sum_array, array_size, experiments_count, 16);
	#endif
	
	#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
	test_vector_add("AVX", &vector_add_avx, x_array, y_array, sum_array, array_size, experiments_count, 32);
	#endif
	
	#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
	const uint64_t aligned_vector_add_avx_aligned_ticks = time_vector_add(&vector_add_avx_aligned, x_array, y_array, sum_array, array_size, experiments_count);
	report_timings("AVX + aligned array", aligned_vector_add_avx_aligned_ticks, array_size);
	#endif
	
	#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
	test_vector_add("AVX + aligned load", &vector_add_avx_load_aligned, x_array, y_array, sum_array, array_size, experiments_count, 32);
	#endif
	
	#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
	test_vector_add("AVX + aligned store", &vector_add_avx_store_aligned, x_array, y_array, sum_array, array_size, experiments_count, 32);
	#endif
	
	printf("Max Method\tAligned CPE\tMin CPE\tMax CPE\n");

	test_vector_max("Naive", &vector_max_naive, x_array, array_size, experiments_count, 16);

	#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
	test_vector_max("SSE2", &vector_max_sse2, x_array, array_size, experiments_count, 16);
	#endif

	#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
	test_vector_max("SSE2 + aligned load", &vector_max_sse2_load_aligned, x_array, array_size, experiments_count, 16);
	#endif

	#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
	test_vector_max("AVX", &vector_max_avx, x_array, array_size, experiments_count, 32);
	#endif

	#ifdef CSE6230_AVX_INTRINSICS_SUPPORTED
	test_vector_max("AVX + aligned load", &vector_max_avx_load_aligned, x_array, array_size, experiments_count, 32);
	#endif

	free(x_array);
	free(y_array);
	free(sum_array);	
}