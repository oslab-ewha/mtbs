#include "mtbs_cu.h"

#include <pthread.h>
#include <unistd.h>

#include <cuda.h>

#include "stream.h"

CUcontext	context;

__device__ tbs_type_t	d_tbs_type;
__device__ skrun_t	*d_skruns;
__device__ unsigned	*d_mtbs_done_cnts;

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

static cudaStream_t	strm_submit;

#define SK_PROTO(name)	__device__ int name(void *args[])
#define SK_FUNCS(base)	SK_PROTO(base);

SK_FUNCS(loopcalc)
SK_FUNCS(mklc)
SK_FUNCS(gma)
SK_FUNCS(lma)
SK_FUNCS(kmeans)
SK_FUNCS(mandelbrot)

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
	default:
		return 0;
	}
}

__device__ void
run_sub_kernel(skrid_t skrid)
{
	skrun_t	*skr;
	int	res;

	skr = &d_skruns[skrid - 1];
	res = run_sub_kernel_func(skr->skid, (void **)skr->args);
	if (get_blockIdxX() == 0 && get_blockIdxY() == 0 && get_threadIdxX() == 0 && get_threadIdxY() == 0) {
		skr->res = res;
	}
}

__global__ void
sub_kernel_func(skrid_t skrid)
{
	run_sub_kernel(skrid);
}

static skrid_t
submit_skrun(skid_t skid, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	skrid_t	skrid;
	skrun_t	skrun;

	skrun.skid = skid;
	skrun.dimGrid = dimGrid;
	skrun.dimBlock = dimBlock;
	memcpy(skrun.args, args, sizeof(void *) * MAX_ARGS);
	skrun.res = 0;
	skrun.n_tbs = dimGrid.x * dimGrid.y;
	skrun.n_mtbs_per_tb = dimBlock.x * dimBlock.y / N_THREADS_PER_mTB;

	pthread_mutex_lock(&mutex);

	skrid = cur_skrid_host + 1;
	info_n_mtbs[skrid - 1] = skrun.n_tbs * skrun.n_mtbs_per_tb;

	cudaMemcpyAsync(g_skruns + cur_skrid_host, &skrun, sizeof(skrun_t), cudaMemcpyHostToDevice, strm_submit);
	cudaStreamSynchronize(strm_submit);

	cur_skrid_host++;

	if (sched->type == TBS_TYPE_SD_STATIC) {
		extern void schedule_mtbs(skrid_t skrid, unsigned n_tbs, unsigned n_mtbs_per_tb);
		schedule_mtbs(skrid, skrun.n_tbs, skrun.n_mtbs_per_tb);
	}

	pthread_mutex_unlock(&mutex);

	return skrid;
}

skrid_t
launch_kernel(skid_t skid, vstream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	skrid_t	skrid;
	cudaStream_t	cstrm = NULL;

	skrid = submit_skrun(skid, dimGrid, dimBlock, args);

	if (sched->type == TBS_TYPE_HW) {
		if (strm != NULL) {
			cstrm = ((vstrm_t)strm)->cudaStrm;
		}
		sub_kernel_func<<<dimGrid, dimBlock, 0, cstrm>>>(skrid);
	}
	return skrid;
}

static void
wait_skrun(skrid_t skrid)
{
	pthread_mutex_lock(&mutex);

	while (!checker_done && !skrun_dones[skrid - 1])
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);
}

void
wait_kernel(skrid_t skrid, vstream_t strm, int *pres)
{
	skrun_t	*skr;
	int	res;

	if (sched->type == TBS_TYPE_HW) {
		if (strm != NULL) {
			cudaStreamSynchronize(((vstrm_t)strm)->cudaStrm);
		}
		else {
			cudaDeviceSynchronize();
		}
	}
	else
		wait_skrun(skrid);

	skr = g_skruns + (skrid - 1);
	cudaMemcpyAsync(&res, &skr->res, sizeof(int), cudaMemcpyDeviceToHost, strm_submit);
	cudaStreamSynchronize(strm_submit);
	*pres = res;
}

static void
notify_done_skruns(unsigned *mtbs_done_cnts, unsigned n_checks)
{
	unsigned	min_new = skrid_done_min;
	BOOL		notify = FALSE;
	unsigned	i;

	pthread_mutex_lock(&mutex);

	for (i = 0; i < n_checks; i++) {
		if (skrun_dones[i + skrid_done_min])
			continue;
		if (mtbs_done_cnts[i] == info_n_mtbs[i + skrid_done_min]) {
			notify = TRUE;
			skrun_dones[i + skrid_done_min] = TRUE;
			if (min_new == i + skrid_done_min) {
				min_new++;
			}
		}
	}
	skrid_done_min = min_new;
	if (notify)
		pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&mutex);
}

static void *
skruns_checkfunc(void *arg)
{
	cudaStream_t	strm;

	cuCtxSetCurrent(context);

	cudaStreamCreate(&strm);

	while (!checker_done) {
		unsigned	n_checks = cur_skrid_host - skrid_done_min;
		if (n_checks > 0) {
			notify_done_skruns(g_mtbs_done_cnts, n_checks);
		}
		usleep(100);
	}

	cudaStreamDestroy(strm);
	return NULL;
}

__global__ void
kernel_init_skrun(tbs_type_t type, skrun_t *skruns, unsigned *mtbs_done_cnts)
{
	int	i;

	d_tbs_type = type;
	d_skruns = skruns;
	d_mtbs_done_cnts = mtbs_done_cnts;
	for (i = 0; i < MAX_QUEUED_KERNELS; i++) {
		skruns[i].skid = 0;
		mtbs_done_cnts[i] = 0;
	}
}

void
init_skrun(void)
{
	cudaError_t	err;

	cuCtxGetCurrent(&context);

	cudaStreamCreate(&strm_submit);

	cudaMalloc(&g_skruns, sizeof(skrun_t) * MAX_QUEUED_KERNELS);
	cudaMallocHost(&g_mtbs_done_cnts, sizeof(unsigned) * MAX_QUEUED_KERNELS);

	info_n_mtbs = (unsigned *)calloc(MAX_QUEUED_KERNELS, sizeof(unsigned));
	skrun_dones = (BOOL *)calloc(MAX_QUEUED_KERNELS, sizeof(BOOL));

	pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	dim3 dimGrid(1,1), dimBlock(1,1);
	kernel_init_skrun<<<dimGrid, dimBlock>>>(sched->type, g_skruns, g_mtbs_done_cnts);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		error("failed to initialize skrun: %s\n", cudaGetErrorString(err));
	else
		cudaDeviceSynchronize();
}

void
fini_skrun(void)
{
	void	*retval;

	checker_done = TRUE;
	pthread_join(checker, &retval);
}
