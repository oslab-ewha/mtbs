#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <math_constants.h>

#include "../../benchapi.h"

__device__ int
irregular(void *args[])
{
	int	n_iters = (int)(long long)args[0];
	int	modbase = (int)(long long)args[1];
	int	midx;
	double	value = 32.192123123213;
	int	i;

	if (get_threadIdxX() % 32 != 0)
		return 0;
	midx = ((get_blockIdxY() * get_gridDimX() + get_blockIdxX()) * get_blockDimX() + get_threadIdxY() * get_blockDimY() + get_threadIdxX()) / 32;
	if (modbase != 0 && midx % modbase != 0)
		return 0;
	for (i = 0; i < n_iters; i++) {
		if (value == CUDART_INF_F)
			value = 329.99128493;
		else
			value = value * 2911.2134324 + 1.992812932;
	}
	return (int)value;
}

int
bench_irregular(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	sk_t	*sks;
	vstream_t	*strms;
	int	count;
	int	res;
	int	i;

	count = (int)(long long)args[0];
	sks = (sk_t *)malloc(sizeof(sk_t) * count);
	strms = (vstream_t *)malloc(sizeof(vstream_t) * count);
	for (i = 0; i < count; i++) {
		strms[i] = create_vstream();
		sks[i] = launch_kernel(IRREGULAR, strms[i], dimGrid, dimBlock, args + 1);
	}

	for (i = 0; i < count; i++)
		wait_kernel(sks[i], strms[i], &res);

	for (i = 0; i < count; i++)
		destroy_vstream(strms[i]);

	free(sks);
	free(strms);

	return res;
}
