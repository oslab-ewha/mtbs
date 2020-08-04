#include "mtbs_cu.h"

extern sched_t	sched_hw;
extern sched_t	sched_sd_dynamic;
extern sched_t	sched_sd_static;
extern sched_t	sched_sd_pagoda;
extern sched_t	sched_sd_gemtc;

static sched_t	*all_sched[] = {
	&sched_hw, &sched_sd_dynamic, &sched_sd_static, &sched_sd_pagoda, &sched_sd_gemtc, NULL
};

sched_t	*sched = &sched_hw;

extern BOOL run_native_tbs(unsigned *pticks);
extern BOOL run_sd_tbs(unsigned *pticks);

extern void assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char skrid);

extern BOOL init_cuda(void);
extern void init_mem(void);
extern void init_skrun(void);
extern void fini_skrun(void);
extern void init_streams(void);
extern void fini_mem(void);

extern "C" void
setup_sched(const char *name)
{
	unsigned	i;

	for (i = 0; all_sched[i]; i++) {
		if (strcmp(name, all_sched[i]->name) == 0) {
			sched = all_sched[i];
			return;
		}
	}

	FATAL(1, "unknown TBS scheduler: %s", name);
}

BOOL
invoke_kernel_func(const char *funcname, void **params)
{
	CUfunction	func_kernel;
	CUstream	strm;
	CUresult	res;

	cuStreamCreate(&strm, CU_STREAM_NON_BLOCKING);

	res = cuModuleGetFunction(&func_kernel, mod, funcname);
	if (res != CUDA_SUCCESS) {
		error("failed to get kernel function: %s: %s", funcname, get_cuda_error_msg(res));
		cuStreamDestroy(strm);
		return FALSE;
	}
	res = cuLaunchKernel(func_kernel, 1, 1, 1, 1, 1, 1, 0, strm, params, NULL);
	if (res != CUDA_SUCCESS) {
		error("failed to initialize skrun: %s\n", get_cuda_error_msg(res));
		cuStreamDestroy(strm);
		return FALSE;
	}

	cuStreamSynchronize(strm);
	cuStreamDestroy(strm);

	return TRUE;
}

extern "C" BOOL
run_tbs(unsigned *pticks)
{
	BOOL	res;

	if (!init_cuda())
		return FALSE;

	init_mem();

	create_fedkern_info();

	init_skrun();
	init_benchruns();
	init_streams();

	if (sched->type == TBS_TYPE_HW)
		res = run_native_tbs(pticks);
	else
		res = run_sd_tbs(pticks);

	fini_skrun();

	free_fedkern_info();

	fini_mem();
	return res;
}
