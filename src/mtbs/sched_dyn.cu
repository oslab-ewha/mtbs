#include "mtbs_cu.h"

#include <pthread.h>

#include "tbs_sd.h"

static skrun_t	*g_skruns;
static BOOL	*g_mtbs_done;

static BOOL	*skrun_dones;

static unsigned	skrid_done_min;
static unsigned	cur_skrid_host;

static BOOL	checker_done;
static pthread_t	checker;

static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	cond = PTHREAD_COND_INITIALIZER;

static CUstream	strm_submit;

unsigned	n_queued_kernels = MAX_QUEUED_KERNELS;

unsigned short	*g_mATs;
unsigned char	*g_mtb_epochs;

#include "sched_dyn.cuh"

static void
notify_done_skruns(unsigned n_checks)
{
	unsigned	min_new = skrid_done_min;
	BOOL		notify = FALSE;
	unsigned	i, idx;

	idx = skrid_done_min;
	for (i = 0; i < n_checks; i++) {
		if (!skrun_dones[idx]) {
			if (g_mtbs_done[idx]) {
				notify = TRUE;
				skrun_dones[idx] = TRUE;
				g_mtbs_done[idx] = FALSE;
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
			notify_done_skruns(n_checks);
		}

		pthread_mutex_unlock(&mutex);
		usleep(100);
	}

	return NULL;
}

static sk_t
submit_skrun_dyn(vstream_t vstream, skrun_t *skr)
{
	skrid_t	skrid;

	pthread_mutex_lock(&mutex);

	while (skrid_done_min == (cur_skrid_host + 1) % n_queued_kernels) {
		/* full */
		pthread_cond_wait(&cond, &mutex);
	}

	skrid = cur_skrid_host + 1;
	skrun_dones[skrid - 1] = FALSE;
	cuMemcpyHtoDAsync((CUdeviceptr)(g_skruns + cur_skrid_host), skr, sizeof(skrun_t), strm_submit);
	/* No synchronization needed */

	cur_skrid_host = (cur_skrid_host + 1) % n_queued_kernels;

	pthread_mutex_unlock(&mutex);

	return (sk_t)(long long)skrid;
}

static void
wait_skrun_dyn(sk_t sk, vstream_t vstream, int *pres)
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
init_skrun_dyn(void)
{
	void	*params[4];
	unsigned	i;

	cuStreamCreate(&strm_submit, CU_STREAM_NON_BLOCKING);

	g_skruns = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t) * n_queued_kernels);
	cuMemAllocHost((void **)&g_mtbs_done, sizeof(BOOL) * n_queued_kernels);
	for (i = 0; i < n_queued_kernels; i++) {
		g_mtbs_done[i] = FALSE;
	}

	skrun_dones = (BOOL *)calloc(n_queued_kernels, sizeof(BOOL));

	pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	g_mATs = (unsigned short *)mtbs_cudaMalloc(EPOCH_MAX * n_max_mtbs * sizeof(unsigned short));
	g_mtb_epochs = (unsigned char *)mtbs_cudaMalloc(n_max_mtbs);

	params[0] = &g_mATs;
	params[1] = &g_mtb_epochs;
	params[2] = &g_skruns;
	params[3] = &g_mtbs_done;
	if (!invoke_kernel_func("setup_sched_dyn", params)) {
		exit(12);
	}
}

static void
fini_skrun_dyn(void)
{
	void	*retval;

	checker_done = TRUE;
	pthread_join(checker, &retval);
	mtbs_cudaFree(g_skruns);

	mtbs_cudaFree(g_mATs);
	mtbs_cudaFree(g_mtb_epochs);
}

sched_t	sched_sd_dynamic = {
	"dynamic",
	TBS_TYPE_SD_DYNAMIC,
	"func_macro_TB_dyn",
	init_skrun_dyn,
	fini_skrun_dyn,
	submit_skrun_dyn,
	wait_skrun_dyn,
};
