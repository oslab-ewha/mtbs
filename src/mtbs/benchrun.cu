#include "mtbs_cu.h"

#include <pthread.h>

extern unsigned	n_submission_workers;

static unsigned	n_readys;

static pthread_mutex_t	worker_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	worker_cond = PTHREAD_COND_INITIALIZER;

static pthread_t	*threads;

static unsigned	n_benches_started;

static void
run_bench(benchrun_t *brun)
{
	bench_func_t	bench;
	int		res;

	bench = brun->info->bench_func;
	res = bench(brun->dimGrid, brun->dimBlock, brun->args);

	brun->res = res;
}

static benchrun_t *
get_benchrun(void)
{
	benchrun_t	*brun = NULL;

	pthread_mutex_lock(&worker_mutex);
	if (n_benches_started < n_benches) {
		brun = benchruns + n_benches_started;
		n_benches_started++;
	}
	pthread_mutex_unlock(&worker_mutex);

	return brun;
}

static void *
worker_func(void *ctx)
{
	cuCtxSetCurrent(context);

	/* wait for start signal */
	pthread_mutex_lock(&worker_mutex);
	n_readys++;
	pthread_cond_wait(&worker_cond, &worker_mutex);
	pthread_mutex_unlock(&worker_mutex);

	while (TRUE) {
		benchrun_t	*brun = get_benchrun();

		if (brun == NULL)
			break;
		run_bench(brun);
	}

	return NULL;
}

void
start_benchruns(void)
{
again:
	pthread_mutex_lock(&worker_mutex);
	if (n_readys < n_submission_workers) {
		pthread_mutex_unlock(&worker_mutex);
		usleep(1);
		goto again;
	}
	pthread_cond_broadcast(&worker_cond);
	pthread_mutex_unlock(&worker_mutex);
}

void
wait_benchruns(void)
{
	int	i;

	for (i = 0; i < n_submission_workers; i++) {
		void	*ret;
		pthread_join(threads[i], &ret);
	}
}

void
init_benchruns(void)
{
	int	i;

	threads = (pthread_t *)malloc(sizeof(pthread_t) * n_submission_workers);
	for (i = 0; i < n_submission_workers; i++) {
		if (pthread_create(&threads[i], NULL, worker_func, NULL) < 0) {
			error("thread creation failed: too many submission workers?");
			exit(10);
		}
	}
	usleep(1000000);
}
