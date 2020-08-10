#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#include "../../benchapi.h"

__device__ int
gma(void *args[])
{
	int	gmemsize = (int)(long long)args[0];
	int	n_iters = (int)(long long)args[1];
	unsigned char	*gmem = (unsigned char *)args[2];
	unsigned	memidx_max = gmemsize * 1024;
	unsigned	randx;
	int	value = 0;
	int	i;

	randx = 0x12345678 + clock() * 19239913 * get_threadIdxX();
	for (i = 0; i < n_iters; i++) {
		unsigned	memidx = randx % memidx_max;
		value += gmem[memidx];
		randx = get_random(randx);
	}
	return value;
}

static int
cookarg_gma(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	unsigned char	*gmem;
	int	gmemsize = (int)(long long)args[0];
	char	buf[1024];
	int	i;

	gmem = (unsigned char *)mtbs_cudaMalloc(gmemsize * 1024);
	if (gmem == NULL) {
		printf("cudaMalloc failed\n");
		return -1;
	}
	for (i = 0; i < 1024; i++) {
		buf[i] = i;
	}
	for (i = 0; i < gmemsize; i++) {
		cuMemcpyHtoD((CUdeviceptr)gmem + i * 1024, buf, 1024);
	}
	args[2] = gmem;
	return 0;
}

int
bench_gma(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	vstream_t	strm;
	sk_t	sk;
	int	res;

	cookarg_gma(dimGrid, dimBlock, args);

	strm = create_vstream();
	sk = launch_kernel(GMA, strm, dimGrid, dimBlock, args);
	wait_kernel(sk, strm, &res);
	destroy_vstream(strm);

	mtbs_cudaFree(args[2]);

	return res;
}
