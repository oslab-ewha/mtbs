#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "../../benchapi.h"

__device__ int
syncsum(void *args[])
{
	int	*data = (int *)args[0];
	int	my_idx = get_threadIdxY() * get_blockDimX() + get_threadIdxX();
	int	sum = 0;

	data[my_idx] = my_idx;
	sync_threads();
	if (get_threadIdxX() == 0 && get_threadIdxY() == 0) {
		unsigned	i;

		for (i = 0; i < get_blockDimX() * get_blockDimY(); i++)
			sum += data[i];
	}
	else
		sum = my_idx;
	sync_threads();
	return sum;
}

int
bench_syncsum(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	vstream_t	strm;
	sk_t	sk;
	int	res;

	args[0] = mtbs_cudaMalloc(sizeof(int) * dimBlock.x * dimBlock.y);
	strm = create_vstream();
	sk = launch_kernel(SYNCSUM, strm, dimGrid, dimBlock, args);
	wait_kernel(sk, strm, &res);
	destroy_vstream(strm);

	mtbs_cudaFree(args[0]);

	return res;
}
