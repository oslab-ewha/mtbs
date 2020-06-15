#include "mtbs_cu.h"

#include <cuda.h>

#include <pthread.h>

#define N_WORKERS	10

extern CUcontext	context;

static pthread_mutex_t	worker_mutex = PTHREAD_MUTEX_INITIALIZER;

static pthread_t	threads[N_WORKERS];

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
	int	i;

	for (i = 0; i < N_WORKERS; i++) {
		pthread_create(&threads[i], NULL, worker_func, NULL);
	}
}

void
wait_benchruns(void)
{
	int	i;

	for (i = 0; i < N_WORKERS; i++) {
		void	*ret;
		pthread_join(threads[i], &ret);
	}
}
