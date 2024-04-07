#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <immintrin.h>
#include <omp.h>

namespace solution {
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
        float *img = new float[num_rows * num_cols];
        float *solution = new float[num_rows * num_cols];
        bitmap_fs.read(reinterpret_cast<char *>(img), sizeof(float) * num_rows * num_cols);
        bitmap_fs.close();

        for (std::int32_t k = 0; k < num_rows * num_cols; k++) {
            int i = k / num_cols, j = k % num_cols;
            if (j == 0 or j == num_cols - 1 or i == 0 or i == num_rows - 1) {
                float sum = 0.0;
                for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) {
                    int ni = i + di, nj = j + dj;
                    if (ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
                        sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
                }
                solution[k] = sum;
                continue;
            }

            int size = j + 16 > num_cols - 1 ? num_cols - j - 1 : 16;
            __m512 sum = _mm512_setzero_ps();
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int ni = i + di, nj = j + dj;
                    __m512 img_v = _mm512_loadu_ps(img + ni * num_cols + nj);
                    __m512 kernel_v = _mm512_set1_ps(kernel[di + 1][dj + 1]);
                    sum = _mm512_fmadd_ps(kernel_v, img_v, sum);
                }
            }
            _mm512_storeu_ps(solution + k, sum);
            k += size - 1;
        }

        sol_fs.write(reinterpret_cast<char *>(solution), sizeof(float) * num_rows * num_cols);
        sol_fs.close();
        return sol_path;
    }
}