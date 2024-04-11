#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,avx512f,avx512dq,avx512cd,avx512bw,avx512vl")
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

#pragma omp parallel
		{
#pragma omp single
			{
				int num_threads = omp_get_num_threads();
				int chunk_size = (num_rows * num_cols) / num_threads;
				int remainder = (num_rows * num_cols) % num_threads;
				int start = 0;
				for (int i = 0; i < num_threads; i++)
				{
					int end = start + chunk_size + (i == num_threads - 1 ? remainder : 0);
#pragma omp task firstprivate(start, end)
					{
						for (int k = start; k < end; k++)
						{
							int i = k / num_cols, j = k % num_cols;

							_mm_prefetch(img + (i + 1) * num_cols + j, _MM_HINT_T0);
							if (j == 0 or j == num_cols - 1 or i == 0 or i == num_rows - 1)
							{
								float sum = 0.0;
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
							if (j + 16 > num_cols - 1)
							{
								float sum = 0.0;
								for (int di = -1; di <= 1; di++)
								{
									for (int dj = -1; dj <= 1; dj++)
									{
										int ni = i + di, nj = j + dj;
										if (ni >= 0 and ni < num_rows and nj >= 0 and nj < num_cols)
											sum += kernel[di + 1][dj + 1] * img[ni * num_cols + nj];
									}
								}
								solution[k] = sum;
							}
							else
							{
								__m512 sum = _mm512_setzero_ps();
								for (int di = -1; di <= 1; di++)
								{
									for (int dj = -1; dj <= 1; dj++)
									{
										int ni = i + di, nj = j + dj;
										__m512 img_v = _mm512_loadu_ps(img + ni * num_cols + nj);
										__m512 kernel_v = _mm512_set1_ps(kernel[di + 1][dj + 1]);
										sum = _mm512_fmadd_ps(kernel_v, img_v, sum);
									}
								}
								_mm512_storeu_ps(solution + k, sum);
								k += 15;
							}
						}
					}
					start = end;
				}
			}
		}
		// munmap(img, sizeof(float) * num_rows * num_cols);
		// munmap(solution, sizeof(float) * num_rows * num_cols);
		// close(bitmap_fd);
		// close(sol_fd);

		return sol_path;
	};
}