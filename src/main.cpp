#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <immintrin.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

namespace solution
{
	std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
	{
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
		int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
		int sol_fd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
		if (bitmap_fd == -1 or sol_fd == -1)
		{
			std::cerr << "Error opening file" << std::endl;
			close(bitmap_fd);
			close(sol_fd);
			exit(EXIT_FAILURE);
		}
		float *img = reinterpret_cast<float *>(mmap(nullptr, sizeof(float) * num_rows * num_cols, PROT_READ, MAP_PRIVATE, bitmap_fd, 0));
		if (img == MAP_FAILED)
		{
			std::cerr << "Error mapping file" << std::endl;
			close(bitmap_fd);
			close(sol_fd);
			exit(EXIT_FAILURE);
		}
		if (ftruncate(sol_fd, sizeof(float) * num_rows * num_cols) == -1)
		{
			std::cerr << "Error truncating file" << std::endl;
			munmap(img, sizeof(float) * num_rows * num_cols);
			close(bitmap_fd);
			close(sol_fd);
			exit(EXIT_FAILURE);
		}

		float *solution = reinterpret_cast<float *>(mmap(nullptr, sizeof(float) * num_rows * num_cols, PROT_READ | PROT_WRITE, MAP_SHARED, sol_fd, 0));

		if (solution == MAP_FAILED)
		{
			std::cerr << "Error mapping file" << std::endl;
			munmap(img, sizeof(float) * num_rows * num_cols);
			close(bitmap_fd);
			close(sol_fd);
			exit(EXIT_FAILURE);
		}

		for (std::int32_t k = 0; k < num_rows * num_cols; k++)
		{
			int i = k / num_cols, j = k % num_cols;
			if (j == 0 or j == num_cols - 1 or i == 0 or i == num_rows - 1)
			{
				float sum = 0.0;
				#pragma omp parallel for
				for (int di = -1; di <= 1; di++)
					for (int dj = -1; dj <= 1; dj++)
					{
						int ni = i + di, nj = j + dj;
						if (ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
							sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
					}
				solution[k] = sum;
				continue;
			}

			int size = j + 8 > num_cols - 1 ? num_cols - j - 1 : 8;
			__m256 sum = _mm256_setzero_ps();
			for (int di = -1; di <= 1; di++)
			{
				for (int dj = -1; dj <= 1; dj++)
				{
					int ni = i + di, nj = j + dj;
					__m256 img_v = _mm256_loadu_ps(img + ni * num_cols + nj);
					__m256 kernel_v = _mm256_set1_ps(kernel[di + 1][dj + 1]);
					sum = _mm256_fmadd_ps(kernel_v, img_v, sum);
				}
			}
			_mm256_storeu_ps(solution + k, sum);
			k += size - 1;
		}

		munmap(img, sizeof(float) * num_rows * num_cols);
		munmap(solution, sizeof(float) * num_rows * num_cols);
		close(bitmap_fd);
		close(sol_fd);

		return sol_path;
	};
}