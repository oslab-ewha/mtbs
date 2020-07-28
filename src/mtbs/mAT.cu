#include "mtbs_cu.h"

#include <pthread.h>

#include "mAT.h"
#include "tbs_sd.h"

static skrun_t	*g_skruns;
static unsigned	*g_mtbs_done_cnts;

static BOOL	*skrun_dones;

static unsigned	skrid_done_min;
static unsigned	cur_skrid_host;

static unsigned	*info_n_mtbs;

static BOOL	checker_done;
static pthread_t	checker;

static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	cond = PTHREAD_COND_INITIALIZER;

static CUstream	strm_submit;

unsigned	n_queued_kernels = MAX_QUEUED_KERNELS;

unsigned short	*g_mATs;
unsigned char	*g_mtb_epochs;

#include "mAT.cuh"

skrid_t
submit_skrun_mAT(skrun_t *skr)
{
	skrid_t	skrid;

	pthread_mutex_lock(&mutex);

	while (skrid_done_min == (cur_skrid_host + 1) % n_queued_kernels) {
		/* full */
		pthread_cond_wait(&cond, &mutex);
	}

	skrid = cur_skrid_host + 1;
	info_n_mtbs[skrid - 1] = skr->n_tbs * skr->n_mtbs_per_tb;
	skrun_dones[skrid - 1] = FALSE;
	cuMemcpyHtoDAsync((CUdeviceptr)(g_skruns + cur_skrid_host), skr, sizeof(skrun_t), strm_submit);
	/* No synchronization needed */

	cur_skrid_host = (cur_skrid_host + 1) % n_queued_kernels;

	pthread_mutex_unlock(&mutex);

	return skrid;
}

void
wait_skrun_mAT(sk_t sk, int *pres)
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
				mtbs_done_cnts[idx] = 0;
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

void
init_skrun_mAT(void)
{
	void	*params[5];

	cuStreamCreate(&strm_submit, CU_STREAM_NON_BLOCKING);

	g_skruns = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t) * n_queued_kernels);
	cuMemAllocHost((void **)&g_mtbs_done_cnts, sizeof(unsigned) * n_queued_kernels);

	info_n_mtbs = (unsigned *)calloc(n_queued_kernels, sizeof(unsigned));
	skrun_dones = (BOOL *)calloc(n_queued_kernels, sizeof(BOOL));

	pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	g_mATs = (unsigned short *)mtbs_cudaMalloc(EPOCH_MAX * n_max_mtbs * sizeof(unsigned short));
	g_mtb_epochs = (unsigned char *)mtbs_cudaMalloc(n_max_mtbs);

	params[0] = &g_mATs;
	params[1] = &g_mtb_epochs;
	params[2] = &n_queued_kernels;
	params[3] = &g_skruns;
	params[4] = &g_mtbs_done_cnts;
	if (!invoke_kernel_func("func_init_skrun_mAT", params)) {
		exit(12);
	}
}

void
fini_skrun_mAT(void)
{
	void	*retval;

	checker_done = TRUE;
	pthread_join(checker, &retval);
	mtbs_cudaFree(g_skruns);

	mtbs_cudaFree(g_mATs);
	mtbs_cudaFree(g_mtb_epochs);
}
