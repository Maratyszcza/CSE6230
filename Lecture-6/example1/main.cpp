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
#ifdef __x86_64__
	uint32_t low, high;
	__asm__ __volatile__ (
		"rdtscp;"
	: "=a"(low), "=d"(high)
	:
	: "%rcx"
	);
	return (uint64_t(high) << 32) | uint64_t(low);
#endif	
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
		if (elapsed_ticks < best_ticks)
			best_ticks = elapsed_ticks;
	}
	return best_ticks;
}

int main(int argc, char** argv) {
	size_t experiments_count = 10000000;
	
	size_t array_size = 500;
	double *x_array = (double*)memalign(16, (array_size + 1) * sizeof(double));
	double *y_array = (double*)memalign(16, (array_size + 1) * sizeof(double));
	double *sum_array = (double*)memalign(16, (array_size + 1) * sizeof(double));
	
	uint64_t min_vector_add_naive_ticks = uint64_t(-1);
	uint64_t max_vector_add_naive_ticks = 0;
	for (size_t x_array_misalignment = 0; x_array_misalignment < 16 / sizeof(double); x_array_misalignment += 1) {
		for (size_t y_array_misalignment = 0; y_array_misalignment < 16 / sizeof(double); y_array_misalignment += 1) {
			for (size_t sum_array_misalignment = 0; sum_array_misalignment < 16 / sizeof(double); sum_array_misalignment += 1) {
				const uint64_t vector_add_naive_ticks = time_vector_add(&vector_add_naive,
					x_array + x_array_misalignment,
					y_array + y_array_misalignment,
					sum_array + sum_array_misalignment,
					array_size, experiments_count);
				min_vector_add_naive_ticks = min(min_vector_add_naive_ticks, vector_add_naive_ticks);
				max_vector_add_naive_ticks = max(max_vector_add_naive_ticks, vector_add_naive_ticks);
			}
		}
	}
	printf("Naive\t%2.2lf CPE\t%2.2lf CPE\n", double(min_vector_add_naive_ticks) / double(array_size), double(max_vector_add_naive_ticks) / double(array_size));
	
	uint64_t min_vector_add_sse2_ticks = uint64_t(-1);
	uint64_t max_vector_add_sse2_ticks = 0;
	for (size_t x_array_misalignment = 0; x_array_misalignment < 16 / sizeof(double); x_array_misalignment += 1) {
		for (size_t y_array_misalignment = 0; y_array_misalignment < 16 / sizeof(double); y_array_misalignment += 1) {
			for (size_t sum_array_misalignment = 0; sum_array_misalignment < 16 / sizeof(double); sum_array_misalignment += 1) {
				const uint64_t vector_add_sse2_ticks = time_vector_add(&vector_add_sse2,
					x_array + x_array_misalignment,
					y_array + y_array_misalignment,
					sum_array + sum_array_misalignment,
					array_size, experiments_count);
				min_vector_add_sse2_ticks = min(min_vector_add_sse2_ticks, vector_add_sse2_ticks);
				max_vector_add_sse2_ticks = max(max_vector_add_sse2_ticks, vector_add_sse2_ticks);
			}
		}
	}
	printf("SSE2\t%2.2lf CPE\t%2.2lf CPE\n", double(min_vector_add_sse2_ticks) / double(array_size), double(max_vector_add_sse2_ticks) / double(array_size));
	
	uint64_t min_vector_add_sse2_load_aligned_ticks = uint64_t(-1);
	uint64_t max_vector_add_sse2_load_aligned_ticks = 0;
	for (size_t x_array_misalignment = 0; x_array_misalignment < 16 / sizeof(double); x_array_misalignment += 1) {
		for (size_t y_array_misalignment = 0; y_array_misalignment < 16 / sizeof(double); y_array_misalignment += 1) {
			for (size_t sum_array_misalignment = 0; sum_array_misalignment < 16 / sizeof(double); sum_array_misalignment += 1) {
				const uint64_t vector_add_sse2_load_aligned_ticks = time_vector_add(&vector_add_sse2_load_aligned,
					x_array + x_array_misalignment,
					y_array + y_array_misalignment,
					sum_array + sum_array_misalignment,
					array_size, experiments_count);
				min_vector_add_sse2_load_aligned_ticks = min(min_vector_add_sse2_load_aligned_ticks, vector_add_sse2_load_aligned_ticks);
				max_vector_add_sse2_load_aligned_ticks = max(max_vector_add_sse2_load_aligned_ticks, vector_add_sse2_load_aligned_ticks);
			}
		}
	}
	printf("SSE2 + aligned load\t%2.2lf CPE\t%2.2lf CPE\n", double(min_vector_add_sse2_load_aligned_ticks) / double(array_size), double(max_vector_add_sse2_load_aligned_ticks) / double(array_size));
	
	uint64_t min_vector_add_sse2_store_aligned_ticks = uint64_t(-1);
	uint64_t max_vector_add_sse2_store_aligned_ticks = 0;
	for (size_t x_array_misalignment = 0; x_array_misalignment < 16 / sizeof(double); x_array_misalignment += 1) {
		for (size_t y_array_misalignment = 0; y_array_misalignment < 16 / sizeof(double); y_array_misalignment += 1) {
			for (size_t sum_array_misalignment = 0; sum_array_misalignment < 16 / sizeof(double); sum_array_misalignment += 1) {
				const uint64_t vector_add_sse2_store_aligned_ticks = time_vector_add(&vector_add_sse2_store_aligned,
					x_array + x_array_misalignment,
					y_array + y_array_misalignment,
					sum_array + sum_array_misalignment,
					array_size, experiments_count);
				min_vector_add_sse2_store_aligned_ticks = min(min_vector_add_sse2_store_aligned_ticks, vector_add_sse2_store_aligned_ticks);
				max_vector_add_sse2_store_aligned_ticks = max(max_vector_add_sse2_store_aligned_ticks, vector_add_sse2_store_aligned_ticks);
			}
		}
	}
	printf("SSE2 + aligned store\t%2.2lf CPE\t%2.2lf CPE\n", double(min_vector_add_sse2_store_aligned_ticks) / double(array_size), double(max_vector_add_sse2_store_aligned_ticks) / double(array_size));
	
	free(x_array);
	free(y_array);
	free(sum_array);	
}