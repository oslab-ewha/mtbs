#include "mtbs_cu.h"

extern void init_sched(void);
extern void fini_sched(void);

extern void wait_fedkern_initialized(fedkern_info_t *d_fkinfo);

extern __device__ void try_setup_dyn_sched(fedkern_info_t *fkinfo);

extern __device__ skrid_t get_skrid_dyn(void);
extern __device__ void advance_epoch_dyn(skrid_t skrid);

__device__ BOOL	going_to_shutdown;

extern "C" __global__ void
func_macro_TB(fedkern_info_t *fkinfo)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		try_setup_dyn_sched(fkinfo);
	}
	__syncthreads();

	while (!going_to_shutdown) {
		skrid_t	skrid;
		skrun_t	*skr;

		skrid = get_skrid_dyn();
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		run_sub_kernel(skr);

		advance_epoch_dyn(skrid);
	}
}

static BOOL
launch_macro_TB(fedkern_info_t *fkinfo)
{
	CUstream	strm;
	CUresult	err;
	CUfunction	func_macro_TB;
	void	*params[1];

	cuStreamCreate(&strm, CU_STREAM_NON_BLOCKING);
	cuModuleGetFunction(&func_macro_TB, mod, "func_macro_TB");

	params[0] = &fkinfo;
	err = cuLaunchKernel(func_macro_TB, n_sm_count, n_MTBs_per_sm, 1,
			     n_threads_per_MTB, 1, 1, 0, strm, params, NULL);
	if (err != CUDA_SUCCESS) {
		error("kernel launch error: %s\n", get_cuda_error_msg(err));
		return FALSE;
	}

	wait_fedkern_initialized(fkinfo);
	return TRUE;
}

static void
stop_macro_TB(fedkern_info_t *fkinfo)
{
	BOOL	done = TRUE;

	cuMemcpyHtoD((CUdeviceptr)&fkinfo->sched_done, &done, sizeof(BOOL));
}

BOOL
run_sd_tbs(unsigned *pticks)
{
	fedkern_info_t	*fkinfo;

	init_sched();

	fkinfo = create_fedkern_info();

	if (!launch_macro_TB(fkinfo))
		return FALSE;

	start_benchruns();

	init_tickcount();

	wait_benchruns();

	*pticks = get_tickcount();

	fini_sched();

	stop_macro_TB(fkinfo);

	free_fedkern_info(fkinfo);

	return TRUE;
}
