#include "mtbs_cu.h"

#include <pthread.h>

#include "tbs_sd.h"
#include "sched_pagoda.h"

#define MAX_ENTRIES_PER_POOL	64

static pthread_spinlock_t	lock;

static tentry_t	*taskentries;
static BOOL	*lock_taskcolumns;
static unsigned	last_column;

static unsigned	n_tasktables;

static tentry_t	*g_taskentries;
static unsigned	numEntriesPerPool;

static __device__ tentry_t	*d_taskentries;
static __device__ int		*d_readytable;
static __device__ unsigned	d_numEntriesPerPool;
static int	*ready_table;
static int	*prev_table;
static int	*next_table;

static CUstream	strm_submit;

#define NEXT_COLUMN(col)	do { (col) = ((col) + 1) % n_tasktables; } while (0)

/* in usec */
static unsigned long long	last_tick_submitted;

#include "sched_pagoda.cuh"

#define N_QUEUED_TASKID_PREV	256

static int	queued_taskid_prevs[N_QUEUED_TASKID_PREV];
static int	qtp_start, qtp_end;

#define NEXT_QTP(qtp)	(((qtp) + 1) % N_QUEUED_TASKID_PREV)

static void
push_prev_taskid(int taskid_prev)
{
	pthread_spin_lock(&lock);

	if (NEXT_QTP(qtp_start) == qtp_end) {
		pthread_spin_unlock(&lock);
		error("queued previous taskid full");
		exit(14);
	}

	queued_taskid_prevs[qtp_start] = taskid_prev;
	qtp_start = NEXT_QTP(qtp_start);
	pthread_spin_unlock(&lock);
}

static int
pop_prev_taskid(int taskid_next)
{
	int	taskid_prev;

	pthread_spin_lock(&lock);
	if (qtp_start == qtp_end) {
		pthread_spin_unlock(&lock);
		return -1;
	}
	taskid_prev = queued_taskid_prevs[qtp_end];
	if (taskid_next < 0) {
		if (prev_table[taskid_prev - 2] > 0) {
			pthread_spin_unlock(&lock);
			return 0;
		}
	}
	qtp_end = NEXT_QTP(qtp_end);

	if (taskid_next > 0) {
		prev_table[taskid_next - 2] = taskid_prev;
		next_table[taskid_prev - 2] = taskid_next;
	}

	pthread_spin_unlock(&lock);

	return taskid_prev;
}

static unsigned
lock_table(void)
{
	unsigned	start_col, col;

again:
	pthread_spin_lock(&lock);
	start_col = last_column;

	while (lock_taskcolumns[last_column]) {
		NEXT_COLUMN(last_column);
		if (last_column == start_col) {
			pthread_spin_unlock(&lock);
			usleep(0);
			goto again;
		}
	}
	lock_taskcolumns[last_column] = TRUE;
	col = last_column;
	NEXT_COLUMN(last_column);
	pthread_spin_unlock(&lock);

	return col;
}

static void
unlock_table(unsigned col)
{
	pthread_spin_lock(&lock);
	lock_taskcolumns[col] = FALSE;
	pthread_spin_unlock(&lock);
}

static void start_dummy_submitter(void);

static void
mark_prev_task_ready(void)
{
	while (TRUE) {
		tentry_t	*d_tentry;
		int	taskid_prev = pop_prev_taskid(-1);

		if (taskid_prev <= 0) {
			if (taskid_prev == 0)
				start_dummy_submitter();
			return;
		}

		d_tentry = g_taskentries + taskid_prev - 2;
		cuMemcpyHtoDAsync((CUdeviceptr)&d_tentry->ready, &taskid_prev, sizeof(int), strm_submit);
		cuStreamSynchronize(strm_submit);
	}
}

static void *
dummy_submitter_func(void *ctx)
{
	while (TRUE) {
		unsigned long long	ticks, ticks_end;

		ticks = get_ticks();

		pthread_spin_lock(&lock);
		ticks_end = last_tick_submitted + 30000;
		if (ticks_end > ticks) {
			pthread_spin_unlock(&lock);
			usleep(ticks_end - ticks);
		}
		else {
			last_tick_submitted = 0;
			pthread_spin_unlock(&lock);

			mark_prev_task_ready();
			break;
		}
	}

	return NULL;
}

static void
start_dummy_submitter(void)
{
	unsigned long long	ticks;

	ticks = get_ticks();

	pthread_spin_lock(&lock);

	if (last_tick_submitted == 0) {
		pthread_t	dummy_submitter;

		last_tick_submitted = ticks;
		pthread_spin_unlock(&lock);

		pthread_create(&dummy_submitter, NULL, dummy_submitter_func, NULL);
		pthread_detach(dummy_submitter);
	}
	else {
		last_tick_submitted = ticks;
		pthread_spin_unlock(&lock);
	}
}

static tentry_t *
find_empty_tentry(unsigned col)
{
	tentry_t	*tentry;
	unsigned	row;

	tentry = taskentries + numEntriesPerPool * col;
	for (row = 0; row < numEntriesPerPool; row++, tentry++) {
		if (tentry->ready == 0)
			return tentry;
	}
	return NULL;
}

static sk_t
submit_skrun_pagoda(vstream_t vstream, skrun_t *skr)
{
	tentry_t	*tentry;
	unsigned	col;
	unsigned	offset;

again:
	col = lock_table();
	tentry = find_empty_tentry(col);
	if (tentry == NULL) {
		unlock_table(col);
		goto again;
	}

	offset = tentry - taskentries;
	tentry->ready = pop_prev_taskid(offset + 2);

	unlock_table(col);

	memcpy(&tentry->skrun, skr, sizeof(skrun_t));
	cuMemcpyHtoDAsync((CUdeviceptr)(g_taskentries + offset), tentry, sizeof(tentry_t), strm_submit);
	cuStreamSynchronize(strm_submit);
	ready_table[offset] = -1;

	push_prev_taskid(offset + 2);

	start_dummy_submitter();

	return (sk_t)tentry;
}

static void
wait_skrun_pagoda(sk_t sk, vstream_t vstream, int *pres)
{
	tentry_t	*tentry = (tentry_t *)sk;
	tentry_t	*d_tentry;
	unsigned	offset;

	offset = tentry - taskentries;
	d_tentry = g_taskentries + offset;
	while (TRUE) {
		if (ready_table[offset] == 0) {
			break;
		}
		usleep(100);
	}

	cuMemcpyDtoHAsync(pres, (CUdeviceptr)&d_tentry->skrun.res, sizeof(int), strm_submit);
	cuStreamSynchronize(strm_submit);
	pthread_spin_lock(&lock);
	if (next_table[offset] > 0) {
		prev_table[next_table[offset] - 2] = 0;
	}
	prev_table[offset] = 0;
	pthread_spin_unlock(&lock);
	tentry->ready = 0;
}

static void
init_skrun_pagoda(void)
{
	void		*params[4];
	unsigned	n_tentries;
	unsigned	i;

	cuStreamCreate(&strm_submit, CU_STREAM_NON_BLOCKING);

	numEntriesPerPool = n_queued_kernels;
	if (numEntriesPerPool > MAX_ENTRIES_PER_POOL)
		numEntriesPerPool = MAX_ENTRIES_PER_POOL;
	n_tasktables = n_sm_count * n_MTBs_per_sm;
	n_tentries = numEntriesPerPool * n_tasktables;
	taskentries = (tentry_t *)malloc(sizeof(tentry_t) * n_tentries);
	lock_taskcolumns = (BOOL *)calloc(n_tasktables, sizeof(BOOL));
	pthread_spin_init(&lock, 0);

	g_taskentries = (tentry_t *)mtbs_cudaMalloc(sizeof(tentry_t) * n_tentries);
	cuMemAllocHost((void **)&ready_table, sizeof(int) * n_tentries);
	prev_table = (int *)malloc(sizeof(int) * n_tentries);
	next_table = (int *)malloc(sizeof(int) * n_tentries);

	for (i = 0; i < n_tentries; i++) {
		taskentries[i].ready = 0;
		taskentries[i].sched = 0;
		ready_table[i] = 0;
		prev_table[i] = 0;
		next_table[i] = 0;
	}

	params[0] = &g_taskentries;
	params[1] = &ready_table;
	params[2] = &numEntriesPerPool;
	params[3] = &n_tasktables;
	invoke_kernel_func("func_init_pagoda", params);
}

static void
fini_skrun_pagoda(void)
{
	cuMemFreeHost(ready_table);
	mtbs_cudaFree(g_taskentries);
}

sched_t sched_sd_pagoda = {
	"pagoda",
	TBS_TYPE_SD_PAGODA,
	"pagoda_master_kernel",
	init_skrun_pagoda,
	fini_skrun_pagoda,
	submit_skrun_pagoda,
	wait_skrun_pagoda,
};
