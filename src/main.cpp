#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <immintrin.h>

namespace solution{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols){
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
		std::ofstream sol_fs(sol_path, std::ios::binary);
		std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
		const auto img = std::make_unique<float[]>(num_rows * num_cols);
		bitmap_fs.read(reinterpret_cast<char*>(img.get()), sizeof(float) * num_rows * num_cols);
		bitmap_fs.close();
		for(std::int32_t k = 0; k < num_rows * num_cols; k++){
				float sum[8] = {0.0};
				int i = k / num_cols, j = k % num_cols;
				if(i == 0 or i == num_rows - 1 or j == 0 or j == num_cols - 1){
					float sum = 0.0;
					for(std::int32_t di = -1; di <= 1; di++)
						for(std::int32_t dj = -1; dj <= 1; dj++) {
							std::int32_t ni = i + di, nj = j + dj;
							if(ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
								sum += kernel[di+1][dj+1] * img[ni * num_cols + nj];
						}
					sol_fs.write(reinterpret_cast<char*>(&sum), sizeof(sum));
					continue;
				}
				if(j+7 < num_cols){
					__m256 sum_v = _mm256_setzero_ps();
					for(std::int32_t di = -1; di <= 1; di++){
						for(std::int32_t dj = -1; dj <= 1; dj++){
							std::int32_t ni = i + di, nj = j + dj;
							if(ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols){
								__m256 img_v = _mm256_loadu_ps(&img[ni * num_cols + nj]);
								__m256 kernel_v = _mm256_set1_ps(kernel[di + 1][dj + 1]);
								sum_v = _mm256_fmadd_ps(img_v, kernel_v, sum_v);
							}
						}
					}
					_mm256_storeu_ps(sum, sum_v);
					for(std::int32_t i = 0; i < 8; i++){
						sol_fs.write(reinterpret_cast<char*>(&sum[i]), sizeof(sum[i]));
					}
					k += 7;
				}
				else{
					float sum = 0.0;
					for(std::int32_t di = -1; di <= 1; di++)
						for(std::int32_t dj = -1; dj <= 1; dj++) {
							std::int32_t ni = i + di, nj = j + dj;
							if(ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
								sum += kernel[di+1][dj+1] * img[ni * num_cols + nj];
						}
					sol_fs.write(reinterpret_cast<char*>(&sum), sizeof(sum));
				}
		}
		sol_fs.close();
		return sol_path;
	}
};