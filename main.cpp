#include <iostream>
#include <nanobench.h>
#include <cstdlib>
#include <immintrin.h>
#include <x86intrin.h>

#define TIMES 0xFFUL

void reduce(const float *array, size_t size, float &result)
{
    result = 0.0;
    for (int i = 0; i < size; ++i)
        result += array[i];
}

void reduce_avx(const float * const array, size_t size, float &result)
{
    // sum
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < size; i += 8)
        sum = _mm256_add_ps(sum, _mm256_load_ps(array + i));
    // horizontal addition
    __m256 temp = _mm256_hadd_ps(sum, sum);
    temp = _mm256_hadd_ps(temp, temp);
    result = _mm256_cvtss_f32(_mm256_add_ps(
            temp, _mm256_permute2f128_ps(temp, temp, 1)));
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    size_t size = (1 << 18);
    auto *array = (float *)aligned_alloc(32, sizeof(float) * size);
    if (!array) {
        std::cerr << "Cannot Alloc Mem" << std::endl;
        exit(1);
    }
    for (int i = 0; i < size; ++i)
        array[i] = 1.0;
    float result;
    // Benchmark Start
    ankerl::nanobench::Bench bench;
    bench.title("Reduce Benchmark")
        .warmup(100)
        .relative(true)
        .performanceCounters(true)
        .minEpochIterations(10);
    bench.run("reduce", [&] {
        for (unsigned long i = 0; i < TIMES; ++i)
            reduce(array, size, result);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
    bench.run("reduce-avx", [&] {
        for (unsigned long i = 0; i < TIMES; ++i)
            reduce_avx(array, size, result);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
    // Benchmark End
    free(array);
    return 0;
}
