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

__global__ void
sub_kernel_func(skrun_t *skr)
{
	run_sub_kernel(skr);
}

static sk_t
submit_skrun(skrun_t *skr)
{
	skrid_t	skrid;

	pthread_mutex_lock(&mutex);

	while (skrid_done_min == (cur_skrid_host + 1) % MAX_QUEUED_KERNELS) {
		/* full */
		pthread_cond_wait(&cond, &mutex);
	}

	skrid = cur_skrid_host + 1;
	info_n_mtbs[skrid - 1] = skr->n_tbs * skr->n_mtbs_per_tb;
	cudaMemcpyAsync(g_skruns + cur_skrid_host, skr, sizeof(skrun_t), cudaMemcpyHostToDevice, strm_submit);
	/* No synchronization needed */

	cur_skrid_host = (cur_skrid_host + 1) % MAX_QUEUED_KERNELS;

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
	cudaStream_t	cstrm = NULL;
	cudaError_t	err;

	if (stream != NULL) {
		cstrm = ((vstrm_t)stream)->cudaStrm;
	}

	d_skr = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t));
	cudaMemcpyAsync(d_skr, skr, sizeof(skrun_t), cudaMemcpyHostToDevice, cstrm);

	sub_kernel_func<<<skr->dimGrid, skr->dimBlock, 0, cstrm>>>(d_skr);
	err = cudaGetLastError();
	if (err != 0) {
		error("hw kernel error: %s", cudaGetErrorString(err));
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

	if (sched->type != TBS_TYPE_HW) {
		sk = submit_skrun(&skrun);
	}
	else {
		sk = submit_skrun_hw(stream, &skrun);
	}
	return sk;
}

static void
wait_skrun(skrid_t skrid)
{
	pthread_mutex_lock(&mutex);

	while (!checker_done && !skrun_dones[skrid - 1])
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);
}

static void
wait_skrun_hw(cudaStream_t cstrm)
{
	if (cstrm != NULL) {
		cudaStreamSynchronize(cstrm);
	}
	else {
		cudaDeviceSynchronize();
	}
}

void
wait_kernel(sk_t sk, vstream_t stream, int *pres)
{
	int	res;

	if (sched->type != TBS_TYPE_HW) {
		skrun_t	*skr;

		skrid_t	skrid = (skrid_t)(long long)sk;
		wait_skrun(skrid);
		skr = g_skruns + (skrid - 1);
		cudaMemcpyAsync(&res, &skr->res, sizeof(int), cudaMemcpyDeviceToHost, strm_submit);
		cudaStreamSynchronize(strm_submit);
	}
	else {
		cudaStream_t	cstrm = NULL;
		skrun_t	*d_skr = (skrun_t *)sk;

		if (stream != NULL)
			cstrm = ((vstrm_t)stream)->cudaStrm;

		cudaMemcpyAsync(&res, &d_skr->res, sizeof(int), cudaMemcpyDeviceToHost, cstrm);
		wait_skrun_hw(cstrm);

		mtbs_cudaFree(d_skr);
	}

	*pres = res;
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
				min_new = (min_new + 1) % MAX_QUEUED_KERNELS;
				notify = TRUE;
			}
		}
		idx = (idx + 1) % MAX_QUEUED_KERNELS;
	}
	skrid_done_min = min_new;
	if (notify)
		pthread_cond_broadcast(&cond);
}

static void *
skruns_checkfunc(void *arg)
{
	cudaStream_t	strm;

	cuCtxSetCurrent(context);

	cudaStreamCreate(&strm);

	while (!checker_done) {
		unsigned	n_checks = (cur_skrid_host + MAX_QUEUED_KERNELS - skrid_done_min) % MAX_QUEUED_KERNELS;
		pthread_mutex_lock(&mutex);

		if (n_checks > 0) {
			notify_done_skruns(g_mtbs_done_cnts, n_checks);
		}

		pthread_mutex_unlock(&mutex);
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

	g_skruns = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t) * MAX_QUEUED_KERNELS);
	cudaMallocHost(&g_mtbs_done_cnts, sizeof(unsigned) * MAX_QUEUED_KERNELS);

	info_n_mtbs = (unsigned *)calloc(MAX_QUEUED_KERNELS, sizeof(unsigned));
	skrun_dones = (BOOL *)calloc(MAX_QUEUED_KERNELS, sizeof(BOOL));

	if (sched->type != TBS_TYPE_HW)
		pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	dim3 dimGrid(1,1), dimBlock(1,1);
	kernel_init_skrun<<<dimGrid, dimBlock>>>(sched->type, g_skruns, g_mtbs_done_cnts);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		error("failed to initialize skrun: %s\n", cudaGetErrorString(err));
		exit(12);
	}
	else
		cudaDeviceSynchronize();
}

void
fini_skrun(void)
{
	void	*retval;

	if (sched->type != TBS_TYPE_HW) {
		checker_done = TRUE;
		pthread_join(checker, &retval);
	}

	mtbs_cudaFree(g_skruns);
}
