#include "mtbs_cu.h"

#include "tbs_sd.h"

static BOOL
launch_macro_TB(void)
{
	CUstream	strm;
	CUfunction	func_macro_TB;
	CUresult	err;

	cuStreamCreate(&strm, CU_STREAM_NON_BLOCKING);

	err = cuModuleGetFunction(&func_macro_TB, mod, sched->macro_TB_funcname);
	if (err != CUDA_SUCCESS) {
		error("failed to get macro kernel function %s: %s\n", sched->macro_TB_funcname, get_cuda_error_msg(err));
		return FALSE;
	}
	err = cuLaunchKernel(func_macro_TB, n_sm_count, n_MTBs_per_sm, 1,
			     n_threads_per_MTB, 1, 1, 0, strm, NULL, NULL);
	if (err != CUDA_SUCCESS) {
		error("kernel launch error: %s\n", get_cuda_error_msg(err));
		return FALSE;
	}

	return TRUE;
}

static void
stop_macro_TB(void)
{
	BOOL	done = TRUE;

	cuMemcpyHtoD((CUdeviceptr)&g_fkinfo->sched_done, &done, sizeof(BOOL));
}

BOOL
run_sd_tbs(unsigned *pticks)
{
	if (!launch_macro_TB())
		return FALSE;

	start_benchruns();

	init_tickcount();

	wait_benchruns();

	*pticks = get_tickcount();

	stop_macro_TB();

	return TRUE;
}
