#include "mtbs_cu.h"

#include "tbs_sd.h"

__device__ benchapi_funcs_t	benchapi_funcs;

__device__ unsigned
get_random(unsigned randx)
{
	randx ^= (randx << 13);
	randx ^= (randx >> 17);
	randx ^= (randx << 5);
	return randx;
}

__device__ int
get_gridDimX(void)
{
        skrun_t    *skr;

	if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return gridDim.x;

	skr = benchapi_funcs.get_skr();

	return skr->dimGrid.x;
}

__device__ int
get_gridDimY(void)
{
        skrun_t	*skr;

        if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return gridDim.y;

	skr = benchapi_funcs.get_skr();

	return skr->dimGrid.y;
}

__device__ int
get_blockIdxX(void)
{
	skrun_t	*skr;
	unsigned	offset;

        if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return blockIdx.x;

	skr = benchapi_funcs.get_skr();
	offset = benchapi_funcs.get_offset_TB();

	return ((offset * N_THREADS_PER_mTB) / (skr->dimBlock.x * skr->dimBlock.y)) % skr->dimGrid.x;
}

__device__ int
get_blockIdxY(void)
{
	skrun_t	*skr;
	unsigned	offset;

        if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return blockIdx.y;

	skr = benchapi_funcs.get_skr();
	offset = benchapi_funcs.get_offset_TB();

	return ((offset * N_THREADS_PER_mTB) / (skr->dimBlock.x * skr->dimBlock.y)) / skr->dimGrid.x;
}

__device__ int
get_blockDimX(void)
{
	skrun_t	*skr;

        if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return blockDim.x;

	skr = benchapi_funcs.get_skr();

	return skr->dimBlock.x;
}

__device__ int
get_blockDimY(void)
{
	skrun_t	*skr;

        if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return blockDim.y;

	skr = benchapi_funcs.get_skr();

	return skr->dimBlock.y;	
}

__device__ int
get_threadIdxX(void)
{
	skrun_t	*skr;
	unsigned	offset;

        if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return threadIdx.x;

	skr = benchapi_funcs.get_skr();
	offset = benchapi_funcs.get_offset_TB();

	return ((offset * N_THREADS_PER_mTB) % (skr->dimBlock.x * skr->dimBlock.y)) % skr->dimBlock.x + (threadIdx.x % N_THREADS_PER_mTB) % skr->dimBlock.x;
}

__device__ int
get_threadIdxY(void)
{
	skrun_t	*skr;
	unsigned	offset;

        if (d_fkinfo->tbs_type == TBS_TYPE_HW)
		return threadIdx.y;

	skr = benchapi_funcs.get_skr();
	offset = benchapi_funcs.get_offset_TB();

	return ((offset * N_THREADS_PER_mTB) % (skr->dimBlock.x * skr->dimBlock.y)) / skr->dimBlock.x + threadIdx.x % N_THREADS_PER_mTB / skr->dimBlock.x;
}

__device__ void
sync_threads(void)
{
        if (d_fkinfo->tbs_type == TBS_TYPE_HW) {
		__syncthreads();
		return;
	}

	benchapi_funcs.sync_TB_threads();
}
