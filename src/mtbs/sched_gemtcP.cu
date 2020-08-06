#include "mtbs_cu.h"

#include <pthread.h>

#include "sched_gemtcP.cuh"

typedef struct {
	skrid_t	skrid;
	unsigned short	id_sm;
} sk_gemtcP_t;

static skrun_t	*g_skruns;
static BOOL	*g_mtbs_dones;

static BOOL	*skrun_dones;

static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	cond = PTHREAD_COND_INITIALIZER;

static unsigned	*skrids_done_min;
static unsigned	*cur_skrids_host;

static BOOL	checker_done;
static pthread_t	checker;

static CUstream	strm_gemtcP;

static unsigned short	id_sm_alloced_last;

static unsigned short
get_submission_id_sm(void)
{
	while (TRUE) {
		unsigned short	id_sm_alloced, id_sm_alloced_start;

		id_sm_alloced = id_sm_alloced_start = id_sm_alloced_last;
		do {
			if (skrids_done_min[id_sm_alloced] != (cur_skrids_host[id_sm_alloced] + 1) % n_queued_kernels) {
				id_sm_alloced_last = (id_sm_alloced + 1) % n_sm_count;
				return id_sm_alloced;
			}
			id_sm_alloced = (id_sm_alloced + 1) % n_sm_count;
		}
		while (id_sm_alloced != id_sm_alloced_start);

		pthread_cond_wait(&cond, &mutex);
	}
}

static sk_t
submit_skrun_gemtcP(vstream_t vstream, skrun_t *skr)
{
	unsigned short	id_sm;
	skrid_t	skrid;
	sk_gemtcP_t	*sk_gemtcP;

	pthread_mutex_lock(&mutex);

	id_sm = get_submission_id_sm();

	skrid = cur_skrids_host[id_sm] + 1;
	skrun_dones[id_sm * n_queued_kernels + skrid - 1] = FALSE;
	cuMemcpyHtoDAsync((CUdeviceptr)(g_skruns + id_sm * n_queued_kernels + cur_skrids_host[id_sm]), skr, sizeof(skrun_t), strm_gemtcP);
	/* No synchronization needed */

	cur_skrids_host[id_sm] = (cur_skrids_host[id_sm] + 1) % n_queued_kernels;

	pthread_mutex_unlock(&mutex);

	sk_gemtcP = (sk_gemtcP_t *)malloc(sizeof(sk_gemtcP_t));
	sk_gemtcP->skrid = skrid;
	sk_gemtcP->id_sm = id_sm;
	return (sk_t)sk_gemtcP;
}

static void
wait_skrun_gemtcP(sk_t sk, vstream_t vstream, int *pres)
{
	skrun_t	*skr;
	sk_gemtcP_t	*sk_gemtcP = (sk_gemtcP_t *)sk;

	pthread_mutex_lock(&mutex);

	while (!checker_done && !skrun_dones[sk_gemtcP->id_sm * n_queued_kernels + sk_gemtcP->skrid - 1])
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);

	skr = g_skruns + sk_gemtcP->id_sm * n_queued_kernels + (sk_gemtcP->skrid - 1);
	cuMemcpyDtoHAsync(pres, (CUdeviceptr)&skr->res, sizeof(int), strm_gemtcP);
	cuStreamSynchronize(strm_gemtcP);

	free(sk_gemtcP);
}

static BOOL
notify_done_skruns(unsigned short id_sm, unsigned n_checks)
{
	unsigned	min_new = skrids_done_min[id_sm];
	BOOL		notify = FALSE;
	unsigned	i, idx;

	idx = skrids_done_min[id_sm];
	for (i = 0; i < n_checks; i++) {
		if (!skrun_dones[id_sm * n_queued_kernels + idx]) {
			if (g_mtbs_dones[id_sm * n_queued_kernels + idx]) {
				notify = TRUE;
				skrun_dones[id_sm * n_queued_kernels + idx] = TRUE;
				g_mtbs_dones[id_sm * n_queued_kernels + idx] = FALSE;
			}
		}
		if (skrun_dones[id_sm * n_queued_kernels + idx]) {
			if (min_new == idx) {
				min_new = (min_new + 1) % n_queued_kernels;
				notify = TRUE;
			}
		}
		idx = (idx + 1) % n_queued_kernels;
	}
	skrids_done_min[id_sm] = min_new;
	return notify;
}

static void *
skruns_checkfunc(void *arg)
{
	while (!checker_done) {
		BOOL	notify = FALSE;

		pthread_mutex_lock(&mutex);

		for (unsigned short i = 0; i < n_sm_count; i++) {
			unsigned	n_checks = (cur_skrids_host[i] + n_queued_kernels - skrids_done_min[i]) % n_queued_kernels;

			if (n_checks > 0) {
				if (notify_done_skruns(i, n_checks))
					notify = TRUE;
			}

		}
		if (notify)
			pthread_cond_broadcast(&cond);
		pthread_mutex_unlock(&mutex);

		usleep(100);
	}

	return NULL;
}

static void
init_skrun_gemtcP(void)
{
	void	*params[2];
	unsigned	i;

	cuStreamCreate(&strm_gemtcP, CU_STREAM_NON_BLOCKING);

	g_skruns = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t) * n_sm_count * n_queued_kernels);
	cuMemAllocHost((void **)&g_mtbs_dones, sizeof(BOOL) * n_sm_count * n_queued_kernels);
	for (i = 0; i < n_sm_count * n_queued_kernels; i++) {
		g_mtbs_dones[i] = FALSE;
	}

	skrids_done_min = (unsigned *)malloc(sizeof(BOOL) * n_sm_count);
	cur_skrids_host = (unsigned *)malloc(sizeof(unsigned) * n_sm_count);
	for (i = 0; i < n_sm_count; i++) {
		skrids_done_min[i] = 0;
		cur_skrids_host[i] = 0;
	}

	skrun_dones = (BOOL *)calloc(n_sm_count * n_queued_kernels, sizeof(BOOL));

	pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	params[0] = &g_skruns;
	params[1] = &g_mtbs_dones;
	if (!invoke_kernel_func("setup_sched_gemtcP", params)) {
		exit(12);
	}
}

static void
fini_skrun_gemtcP(void)
{
	void	*retval;

	checker_done = TRUE;
	pthread_join(checker, &retval);
	mtbs_cudaFree(g_skruns);
}

sched_t	sched_sd_gemtcP = {
	"gemtcP",
	TBS_TYPE_SD_GEMTCP,
	"func_macro_TB_gemtcP",
	init_skrun_gemtcP,
	fini_skrun_gemtcP,
	submit_skrun_gemtcP,
	wait_skrun_gemtcP,
};
