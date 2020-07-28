#include "mtbs_cu.h"

extern sched_t	sched_hw;
extern sched_t	sched_sd_dynamic;
extern sched_t	sched_sd_static;
extern sched_t	sched_sd_pagoda;

static sched_t	*all_sched[] = {
	&sched_hw, &sched_sd_dynamic, &sched_sd_static, &sched_sd_pagoda, NULL
};

sched_t	*sched = &sched_hw;
unsigned	sched_id = 1;
char		*sched_argstr;

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
setup_sched(const char *strpol)
{
	unsigned	i;

	for (i = 0; all_sched[i]; i++) {
		int	len = strlen(all_sched[i]->name);

		if (strncmp(strpol, all_sched[i]->name, len) == 0) {
			tbs_type_t	type;

			sched = all_sched[i];
			type = sched->type;
			sched_id = i + 1;

			if (strpol[len] ==':')
				sched_argstr = strdup(strpol + len + 1);
			else if (strpol[len] != '\0')
				continue;

			sched->name = strdup(strpol);
			sched->type = type;
			return;
		}
	}

	FATAL(1, "unknown scheduling policy: %s", strpol);
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

	if (sched->type != TBS_TYPE_HW)
		create_fedkern_info();

	init_skrun();
	init_benchruns();
	init_streams();

	if (sched->type == TBS_TYPE_HW)
		res = run_native_tbs(pticks);
	else
		res = run_sd_tbs(pticks);

	fini_skrun();

	if (sched->type != TBS_TYPE_HW)
		free_fedkern_info();

	fini_mem();
	return res;
}
