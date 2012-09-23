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
#if 0
	uint32_t low, high;
	__asm__ __volatile__ (
		"rdtscp;"
	: "=a"(low), "=d"(high)
	:
	: "%rcx"
	);
	return (uint64_t(high) << 32) | uint64_t(low);
#else
	return get_cpu_ticks_start();
#endif	
}

inline static uint64_t min(uint64_t a, uint64_t b) {
	return a < b ? a : b;
}

inline static uint64_t max(uint64_t a, uint64_t b) {
	return a > b ? a : b;
}

static uint64_t time_dot_product(vector3d_dot_products_function vector3d_dot_products, const double* v_vectors, const double* u_vectors, double* dp_array, size_t vectors_count, size_t experiments_count) {
	uint64_t best_ticks = uint64_t(-1);
	for (size_t experiment_number = 1; experiment_number <= experiments_count; experiment_number++) {
		const uint64_t start_ticks = get_cpu_ticks_start();
		vector3d_dot_products(v_vectors, u_vectors, dp_array, vectors_count);
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

static void report_timings(const char* method_name, uint64_t aligned_ticks, size_t array_size) {
	printf("%20s\t%2.2lf\n", method_name, double(aligned_ticks) / double(array_size));
}

int main(int argc, char** argv) {
	size_t experiments_count = 100000;
	
	size_t vectors_count = 150;
	const size_t components_per_vector = 3;
	double *v_vectors = (double*)memalign(32, vectors_count * components_per_vector * sizeof(double) + 32);
	double *u_vectors = (double*)memalign(32, vectors_count * components_per_vector * sizeof(double) + 32);
	double *dp_array = (double*)memalign(32, vectors_count * sizeof(double) + 32);
	
	printf("Method\tAligned CPE\tMin CPE\tMax CPE\n");
	
	const uint64_t aligned_vector3d_dot_products_naive_ticks = time_dot_product(&vector3d_dot_products_naive, v_vectors, u_vectors, dp_array, vectors_count, experiments_count);
	uint64_t min_vector3d_dot_products_naive_ticks = uint64_t(-1);
	uint64_t max_vector3d_dot_products_naive_ticks = 0;
	for (size_t v_pointer_misalignment = 0; v_pointer_misalignment < 16 / sizeof(double); v_pointer_misalignment += 1) {
		for (size_t u_pointer_misalignment = 0; u_pointer_misalignment < 16 / sizeof(double); u_pointer_misalignment += 1) {
			for (size_t dp_array_misalignment = 0; dp_array_misalignment < 16 / sizeof(double); dp_array_misalignment += 1) {
				const uint64_t vector3d_dot_products_naive_ticks = time_dot_product(&vector3d_dot_products_naive,
					v_vectors + v_pointer_misalignment,
					u_vectors + u_pointer_misalignment,
					dp_array + dp_array_misalignment,
					vectors_count, experiments_count);
				min_vector3d_dot_products_naive_ticks = min(min_vector3d_dot_products_naive_ticks, vector3d_dot_products_naive_ticks);
				max_vector3d_dot_products_naive_ticks = max(max_vector3d_dot_products_naive_ticks, vector3d_dot_products_naive_ticks);
			}
		}
	}
	report_timings("Naive", aligned_vector3d_dot_products_naive_ticks, min_vector3d_dot_products_naive_ticks, max_vector3d_dot_products_naive_ticks, vectors_count);
	
	#ifdef CSE6230_SSE2_INTRINSICS_SUPPORTED
	const uint64_t aligned_vector3d_dot_products_sse2_ticks = time_dot_product(&vector3d_dot_products_sse2, v_vectors, u_vectors, dp_array, vectors_count, experiments_count);
	uint64_t min_vector3d_dot_products_sse2_ticks = uint64_t(-1);
	uint64_t max_vector3d_dot_products_sse2_ticks = 0;
	for (size_t v_pointer_misalignment = 0; v_pointer_misalignment < 16 / sizeof(double); v_pointer_misalignment += 1) {
		for (size_t u_pointer_misalignment = 0; u_pointer_misalignment < 16 / sizeof(double); u_pointer_misalignment += 1) {
			for (size_t dp_array_misalignment = 0; dp_array_misalignment < 16 / sizeof(double); dp_array_misalignment += 1) {
				const uint64_t vector3d_dot_products_sse2_ticks = time_dot_product(&vector3d_dot_products_sse2,
					v_vectors + v_pointer_misalignment,
					u_vectors + u_pointer_misalignment,
					dp_array + dp_array_misalignment,
					vectors_count, experiments_count);
				min_vector3d_dot_products_sse2_ticks = min(min_vector3d_dot_products_sse2_ticks, vector3d_dot_products_sse2_ticks);
				max_vector3d_dot_products_sse2_ticks = max(max_vector3d_dot_products_sse2_ticks, vector3d_dot_products_sse2_ticks);
			}
		}
	}
	report_timings("SSE2", aligned_vector3d_dot_products_sse2_ticks, min_vector3d_dot_products_sse2_ticks, max_vector3d_dot_products_sse2_ticks, vectors_count);
	#endif
	
	#ifdef CSE6230_SSE3_INTRINSICS_SUPPORTED
	const uint64_t aligned_vector3d_dot_products_sse3_ticks = time_dot_product(&vector3d_dot_products_sse3, v_vectors, u_vectors, dp_array, vectors_count, experiments_count);
	uint64_t min_vector3d_dot_products_sse3_ticks = uint64_t(-1);
	uint64_t max_vector3d_dot_products_sse3_ticks = 0;
	for (size_t v_pointer_misalignment = 0; v_pointer_misalignment < 16 / sizeof(double); v_pointer_misalignment += 1) {
		for (size_t u_pointer_misalignment = 0; u_pointer_misalignment < 16 / sizeof(double); u_pointer_misalignment += 1) {
			for (size_t dp_array_misalignment = 0; dp_array_misalignment < 16 / sizeof(double); dp_array_misalignment += 1) {
				const uint64_t vector3d_dot_products_sse3_ticks = time_dot_product(&vector3d_dot_products_sse3,
					v_vectors + v_pointer_misalignment,
					u_vectors + u_pointer_misalignment,
					dp_array + dp_array_misalignment,
					vectors_count, experiments_count);
				min_vector3d_dot_products_sse3_ticks = min(min_vector3d_dot_products_sse3_ticks, vector3d_dot_products_sse3_ticks);
				max_vector3d_dot_products_sse3_ticks = max(max_vector3d_dot_products_sse3_ticks, vector3d_dot_products_sse3_ticks);
			}
		}
	}
	report_timings("SSE3", aligned_vector3d_dot_products_sse3_ticks, min_vector3d_dot_products_sse3_ticks, max_vector3d_dot_products_sse3_ticks, vectors_count);
	#endif

	free(v_vectors);
	free(u_vectors);
	free(dp_array);	
}