#include "mtbs_cu.h"

#include <unistd.h>

#include "stream.h"

CUcontext	context;

CUfunction	func_sub_kernel;

__device__ tbs_type_t	d_tbs_type;
__device__ skrun_t	*d_skruns;
__device__ unsigned	dn_queued_kernels;

#define SK_PROTO(name)	__device__ int name(void *args[])
#define SK_FUNCS(base)	SK_PROTO(base);

SK_FUNCS(loopcalc)
SK_FUNCS(mklc)
SK_FUNCS(gma)
SK_FUNCS(lma)
SK_FUNCS(kmeans)
SK_FUNCS(mandelbrot)
SK_FUNCS(irregular)
SK_FUNCS(mm)

static __device__ int
run_sub_kernel_func(skid_t skid, void *args[])
{
	switch (skid) {
	case LOOPCALC:
		return loopcalc(args);
	case MKLC:
		return mklc(args);		
	case GMA:
		return gma(args);
	case LMA:
		return lma(args);
	case KMEANS:
		return kmeans(args);
	case MANDELBROT:
		return mandelbrot(args);
	case IRREGULAR:
		return irregular(args);
	case MM:
		return mm(args);
	default:
		return 0;
	}
}

__device__ void
run_sub_kernel(skrun_t *skr)
{
	int	res;

	res = run_sub_kernel_func(skr->skid, (void **)skr->args);
	if (get_blockIdxX() == 0 && get_blockIdxY() == 0 && get_threadIdxX() == 0 && get_threadIdxY() == 0) {
		skr->res = res;
	}
}

extern "C" __global__ void
sub_kernel_func(skrun_t *skr)
{
	run_sub_kernel(skr);
}

sk_t
launch_kernel(skid_t skid, vstream_t stream, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	skrun_t	skrun;

	skrun.skid = skid;
	skrun.dimGrid = dimGrid;
	skrun.dimBlock = dimBlock;
	memcpy(skrun.args, args, sizeof(void *) * MAX_ARGS);
	skrun.res = 0;
	skrun.n_tbs = dimGrid.x * dimGrid.y;
	skrun.n_mtbs_per_tb = dimBlock.x * dimBlock.y / N_THREADS_PER_mTB;

	return sched->submit_skrun(stream, &skrun);
}

void
wait_kernel(sk_t sk, vstream_t vstream, int *pres)
{
	sched->wait_skrun(sk, vstream, pres);
}

extern "C" __global__ void
func_init_skrun(tbs_type_t type)
{
	d_tbs_type = type;
}

void
init_skrun(void)
{
	CUfunction	func_init_skrun;
	void		*params[1];
	CUstream	strm;
	CUresult res;

	res = cuModuleGetFunction(&func_sub_kernel, mod, "sub_kernel_func");
	if (res != CUDA_SUCCESS) {
		error("failed to get sub_kernel_func: %s\n", get_cuda_error_msg(res));
	}

	if (sched->init_skrun)
		sched->init_skrun();

	cuModuleGetFunction(&func_init_skrun, mod, "func_init_skrun");
	params[0] = &sched->type;
	cuStreamCreate(&strm, CU_STREAM_NON_BLOCKING);
	cuLaunchKernel(func_init_skrun, 1, 1, 1, 1, 1, 1, 0, strm, params, NULL);
	cuStreamDestroy(strm);
}

void
fini_skrun(void)
{
	if (sched->fini_skrun)
		sched->fini_skrun();
}
