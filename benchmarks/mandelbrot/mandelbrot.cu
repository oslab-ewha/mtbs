// Note: Most of the code comes from the MacResearch OpenCL podcast
#include <stdio.h>

#include "../benchapi.h"

__device__ int
mandelbrot(void *args[])
{
	char	*out = (char *)args[0];
	int	width = (int)(long long)args[1];
	int	mx = (int)(long long)args[2];
	int	my = (int)(long long)args[3];
	int	x_dim, y_dim;
	int	res = 0;
	int	i, j;

	x_dim = (get_blockIdxX() * get_blockDimX() + get_threadIdxX()) * mx;
	y_dim = (get_blockIdxY() * get_blockDimY() + get_threadIdxY()) * my;

	for (i = 0; i < mx; i++) {
		for (j = 0; j < my; j++) {
			int	index = 3 * width * (y_dim + j) + (x_dim + i) * 3;
			float	x_origin = ((float)(x_dim + i)/ width) * 3.25 - 2;
			float	y_origin = ((float)(y_dim + j)/ width) * 2.5 - 1.25;

			float x = 0.0;
			float y = 0.0;

			int	iteration = 0;
			int	max_iteration = 2048;
			while (x * x + y * y <= 14 && iteration < max_iteration) {
				float xtemp = x * x - y * y + x_origin;
				y = 2 * x * y + y_origin;
				x = xtemp;
				iteration++;
			}

			if (iteration == max_iteration) {
				out[index] = 0;
				out[index + 1] = 0;
				out[index + 2] = 0;
			} else {
				out[index] = iteration;
				out[index + 1] = iteration;
				out[index + 2] = iteration;
			}
			res = iteration;
		}
	}

	return res;
}

int
bench_mandelbrot(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	int	width, height, mx, my;
	char	*image, *host_image;
	size_t	buffer_size;
	int		res;
	skrid_t		skrid;
	vstream_t	strm;

	width = (int)(long long)args[0];
	height = (int)(long long)args[1];

	mx = width / (dimGrid.x * dimBlock.x);
	my = height / (dimGrid.y * dimBlock.y);

	// Multiply by 3 here, since we need red, green and blue for each pixel
	buffer_size = sizeof(char) * width * height * 3;

	if (cudaMalloc((void **)&image, buffer_size) != cudaSuccess) {
		printf("failed to allocate mem: buffer size: %ld\n", buffer_size);
		return -1;
	}
	host_image = (char *)malloc(buffer_size);

	args[0] = image;
	args[1] = (void *)(long long)width;
	args[2] = (void *)(long long)mx;
	args[3] = (void *)(long long)my;

	strm = create_vstream();
	skrid = launch_kernel(MANDELBROT, strm, dimGrid, dimBlock, args);
	wait_kernel(skrid, strm, &res);

	cudaMemcpyAsync(host_image, image, buffer_size, cudaMemcpyDeviceToHost, *(cudaStream_t *)strm);
	cudaStreamSynchronize(*(cudaStream_t *)strm);

	destroy_vstream(strm);

#if 0
	//TODO: cudaFree freeze the system
	cudaFree(image);
#endif
	free(host_image);

	return res;
}
