#include "mtbs_cu.h"

#include <pthread.h>
#include <unistd.h>

#include "stream.h"

CUcontext	context;

static CUfunction	func_sub_kernel;

unsigned	n_queued_kernels = MAX_QUEUED_KERNELS;

__device__ tbs_type_t	d_tbs_type;
__device__ skrun_t	*d_skruns;
__device__ unsigned	*d_mtbs_done_cnts;
__device__ unsigned	dn_queued_kernels;

static skrun_t	*g_skruns;
static unsigned	*g_mtbs_done_cnts;

static unsigned	*info_n_mtbs;

static BOOL	*skrun_dones;
static unsigned	skrid_done_min;
static unsigned	cur_skrid_host;

static BOOL	checker_done;
static pthread_t	checker;
static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	cond = PTHREAD_COND_INITIALIZER;

CUstream	strm_submit;

#define SK_PROTO(name)	__device__ int name(void *args[])
#define SK_FUNCS(base)	SK_PROTO(base);

extern void init_skrun_pagoda(void);
extern void fini_skrun_pagoda(void);
extern sk_t submit_skrun_pagoda(skrun_t *skr);
extern void wait_skrun_pagoda(sk_t sk, int *pres);

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

static sk_t
submit_skrun(skrun_t *skr)
{
	skrid_t	skrid;

	pthread_mutex_lock(&mutex);

	while (skrid_done_min == (cur_skrid_host + 1) % n_queued_kernels) {
		/* full */
		pthread_cond_wait(&cond, &mutex);
	}

	skrid = cur_skrid_host + 1;
	info_n_mtbs[skrid - 1] = skr->n_tbs * skr->n_mtbs_per_tb;
	cuMemcpyHtoDAsync((CUdeviceptr)(g_skruns + cur_skrid_host), skr, sizeof(skrun_t), strm_submit);
	/* No synchronization needed */

	cur_skrid_host = (cur_skrid_host + 1) % n_queued_kernels;

	if (sched->type == TBS_TYPE_SD_STATIC) {
		extern void schedule_mtbs(skrid_t skrid, unsigned n_tbs, unsigned n_mtbs_per_tb);
		schedule_mtbs(skrid, skr->n_tbs, skr->n_mtbs_per_tb);
	}

	pthread_mutex_unlock(&mutex);

	return (sk_t)(long long)skrid;
}

static sk_t
submit_skrun_hw(vstream_t stream, skrun_t *skr)
{
	skrun_t	*d_skr;
	CUstream	cstrm = NULL;
	void	*params[1];
	CUresult	err;

	if (stream != NULL) {
		cstrm = ((vstrm_t)stream)->cudaStrm;
	}

	d_skr = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t));
	cuMemcpyHtoDAsync((CUdeviceptr)d_skr, skr, sizeof(skrun_t), cstrm);

	params[0] = &d_skr;
	err = cuLaunchKernel(func_sub_kernel,
			     skr->dimGrid.x, skr->dimGrid.y, skr->dimGrid.z,
			     skr->dimBlock.x, skr->dimBlock.y, skr->dimBlock.z,
			     0, cstrm, params, NULL);
	if (err != CUDA_SUCCESS) {
		error("hw kernel error: %s", get_cuda_error_msg(err));
	}
	return d_skr;
}

sk_t
launch_kernel(skid_t skid, vstream_t stream, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	skrun_t	skrun;
	sk_t	sk;

	skrun.skid = skid;
	skrun.dimGrid = dimGrid;
	skrun.dimBlock = dimBlock;
	memcpy(skrun.args, args, sizeof(void *) * MAX_ARGS);
	skrun.res = 0;
	skrun.n_tbs = dimGrid.x * dimGrid.y;
	skrun.n_mtbs_per_tb = dimBlock.x * dimBlock.y / N_THREADS_PER_mTB;

	switch (sched->type) {
	case TBS_TYPE_SD_DYNAMIC:
	case TBS_TYPE_SD_STATIC:
		sk = submit_skrun(&skrun);
		break;
	case TBS_TYPE_SD_PAGODA:
		sk = submit_skrun_pagoda(&skrun);
		break;
	default:
		sk = submit_skrun_hw(stream, &skrun);
		break;
	}
	return sk;
}

static void
wait_skrun(sk_t sk, int *pres)
{
	skrun_t	*skr;

	skrid_t	skrid = (skrid_t)(long long)sk;

	pthread_mutex_lock(&mutex);

	while (!checker_done && !skrun_dones[skrid - 1])
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);

	skr = g_skruns + (skrid - 1);
	cuMemcpyDtoHAsync(pres, (CUdeviceptr)&skr->res, sizeof(int), strm_submit);
	cuStreamSynchronize(strm_submit);
}

static void
wait_skrun_hw(sk_t sk, vstream_t stream, int *pres)
{
	CUstream	cstrm;
	skrun_t	*d_skr = (skrun_t *)sk;

	cstrm = ((vstrm_t)stream)->cudaStrm;

	cuStreamSynchronize(cstrm);
	cuMemcpyDtoHAsync(pres, (CUdeviceptr)&d_skr->res, sizeof(int), cstrm);
	cuStreamSynchronize(cstrm);

	mtbs_cudaFree(d_skr);
}

void
wait_kernel(sk_t sk, vstream_t stream, int *pres)
{
	switch (sched->type) {
	case TBS_TYPE_SD_DYNAMIC:
	case TBS_TYPE_SD_STATIC:
		wait_skrun(sk, pres);
		break;
	case TBS_TYPE_SD_PAGODA:
		wait_skrun_pagoda(sk, pres);
		break;
	default:
		wait_skrun_hw(sk, stream, pres);
		break;
	}
}

static void
notify_done_skruns(unsigned *mtbs_done_cnts, unsigned n_checks)
{
	unsigned	min_new = skrid_done_min;
	BOOL		notify = FALSE;
	unsigned	i, idx;

	idx = skrid_done_min;
	for (i = 0; i < n_checks; i++) {
		if (!skrun_dones[idx]) {
			if (mtbs_done_cnts[idx] == info_n_mtbs[idx]) {
				notify = TRUE;
				skrun_dones[idx] = TRUE;
			}
		}
		if (skrun_dones[idx]) {
			if (min_new == idx) {
				min_new = (min_new + 1) % n_queued_kernels;
				notify = TRUE;
			}
		}
		idx = (idx + 1) % n_queued_kernels;
	}
	skrid_done_min = min_new;
	if (notify)
		pthread_cond_broadcast(&cond);
}

static void *
skruns_checkfunc(void *arg)
{
	while (!checker_done) {
		unsigned	n_checks = (cur_skrid_host + n_queued_kernels - skrid_done_min) % n_queued_kernels;
		pthread_mutex_lock(&mutex);

		if (n_checks > 0) {
			notify_done_skruns(g_mtbs_done_cnts, n_checks);
		}

		pthread_mutex_unlock(&mutex);
		usleep(100);
	}

	return NULL;
}

extern "C" __global__ void
func_init_skrun(tbs_type_t type, unsigned n_queued_kernels, skrun_t *skruns, unsigned *mtbs_done_cnts)
{
	int	i;

	d_tbs_type = type;
	dn_queued_kernels = n_queued_kernels;
	d_skruns = skruns;
	d_mtbs_done_cnts = mtbs_done_cnts;
	for (i = 0; i < dn_queued_kernels; i++) {
		skruns[i].skid = 0;
		mtbs_done_cnts[i] = 0;
	}
}

void
init_skrun(void)
{
	CUfunction	func_init_skrun;
	void		*params[4];
	CUresult	err;

	cuStreamCreate(&strm_submit, CU_STREAM_NON_BLOCKING);

	g_skruns = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t) * n_queued_kernels);
	cuMemAllocHost((void **)&g_mtbs_done_cnts, sizeof(unsigned) * n_queued_kernels);

	info_n_mtbs = (unsigned *)calloc(n_queued_kernels, sizeof(unsigned));
	skrun_dones = (BOOL *)calloc(n_queued_kernels, sizeof(BOOL));

	if (sched->type != TBS_TYPE_HW && sched->type != TBS_TYPE_SD_PAGODA)
		pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	cuModuleGetFunction(&func_init_skrun, mod, "func_init_skrun");
	params[0] = &sched->type;
	params[1] = &n_queued_kernels;
	params[2] = &g_skruns;
	params[3] = &g_mtbs_done_cnts;
	err = cuLaunchKernel(func_init_skrun, 1, 1, 1, 1, 1, 1, 0, strm_submit, params, NULL);
	if (err != CUDA_SUCCESS) {
		error("failed to initialize skrun: %s\n", get_cuda_error_msg(err));
		exit(12);
	}
	cuStreamSynchronize(strm_submit);

	CUresult res = cuModuleGetFunction(&func_sub_kernel, mod, "sub_kernel_func");
	if (res != CUDA_SUCCESS) {
		error("failed to get sub_kernel_func: %s\n", get_cuda_error_msg(res));
	}
	if (sched->type == TBS_TYPE_SD_PAGODA)
		init_skrun_pagoda();
}

void
fini_skrun(void)
{
	void	*retval;

	switch (sched->type) {
	case TBS_TYPE_SD_DYNAMIC:
	case TBS_TYPE_SD_STATIC:
		checker_done = TRUE;
		pthread_join(checker, &retval);
	case TBS_TYPE_SD_PAGODA:
		fini_skrun_pagoda();
		break;
	default:
		break;
	}

	mtbs_cudaFree(g_skruns);
}
