#include <stdio.h>

#include "../benchapi.h"

__device__ int
mm(void *args[])
{
	int	m = (int)(long long)args[0];
	int	n = (int)(long long)args[1];
	int	k = (int)(long long)args[2];
	int	*mat = (int *)args[3], *matA, *matB, *matC;
	int	mx, my;
	int	res = 1;
	int	row, col;

	mx = m / (get_gridDimX() * get_blockDimX());
	my = k / (get_gridDimY() * get_blockDimY());

	matA = mat;
	matB = matA + m * n;
	matC = matB + n * k;

	row = (get_blockIdxY() * get_blockDimY() + get_threadIdxY()) * my;
	col = (get_blockIdxX() * get_blockDimX() + get_threadIdxX()) * mx;

	for (int x = 0; x < mx; x++) {
		for (int y = 0; y < my; y++) {
			int	sum = 0;
			for (int i = 0; i < n; i++) {
				sum += matA[(row + y) * n + i] * matB[i * k + col + x];
			}
			matC[(row + y) * k + col + x] = sum;
			res = sum;
		}
	}
	return res;
}

static void
fill_matrix(int *mat, int x, int y)
{
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			mat[i * x + j] = ((i + 12) * (j + 3)) % 1024;
		}
	}
}

int
bench_mm(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	int	m, n, k;
	int	*matA, *matB, *matC;
	int	*d_mat;
	int		res;
	sk_t		sk;
	vstream_t	strm;

	m = (int)(long long)args[0];
	n = (int)(long long)args[1];
	k = (int)(long long)args[2];

	matA = (int *)malloc(sizeof(int) * m * n);
	matB = (int *)malloc(sizeof(int) * n * k);
	matC = (int *)malloc(sizeof(int) * m * k);

	fill_matrix(matA, m, n);
	fill_matrix(matB, n, k);

	d_mat = (int *)mtbs_cudaMalloc(sizeof(int) * (m * n + n * k + m * k));
	if (d_mat == NULL) {
		printf("failed to allocate mem: buffer size: %ld\n", sizeof(int) * m * n + sizeof(int) * n * k);
		return -1;
	}

	args[0] = (void *)(long long)m;
	args[1] = (void *)(long long)n;
	args[2] = (void *)(long long)k;
	args[3] = d_mat;

	strm = create_vstream();

	cudaMemcpyAsync(d_mat, matA, sizeof(int) * m * n, cudaMemcpyHostToDevice, *(cudaStream_t *)strm);
	cudaMemcpyAsync(d_mat + m * n, matB, sizeof(int) * n * k, cudaMemcpyHostToDevice, *(cudaStream_t *)strm);
	cudaStreamSynchronize(*(cudaStream_t *)strm);

	sk = launch_kernel(MM, strm, dimGrid, dimBlock, args);
	wait_kernel(sk, strm, &res);

	cudaMemcpyAsync(matC, d_mat + (m * n + n * k), sizeof(int) * (m * k), cudaMemcpyDeviceToHost, *(cudaStream_t *)strm);
	cudaStreamSynchronize(*(cudaStream_t *)strm);

	destroy_vstream(strm);

	mtbs_cudaFree(d_mat);

	free(matA);
	free(matB);
	free(matC);

	return res;
}
