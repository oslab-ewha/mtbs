typedef struct {
	unsigned	exec;
	int	taskid;
	unsigned short	offset;
} wentry_t;

static __shared__ wentry_t	warptable[31];
static __shared__ unsigned	doneCtr[MAX_ENTRIES_PER_POOL];

static __device__ BOOL	*psched_done;

__device__ skrun_t *
get_skr_pagoda(void)
{
	wentry_t	*wentry = &warptable[threadIdx.x / 32 - 1];
	tentry_t	*tentry = d_taskentries + wentry->taskid - 2;

	return &tentry->skrun;
}

__device__ unsigned short
get_offset_TB_pagoda(void)
{
	wentry_t	*wentry = &warptable[threadIdx.x / 32 - 1];
	return wentry->offset;
}

static __device__ void
assign_task(int taskid, unsigned count)
{
	unsigned	i;
	unsigned	offset = 0;

again:
	for (i = 0; i < 31; i++) {
		if (!atomicCAS(&warptable[i].exec, 0, 1)) {
			warptable[i].taskid = taskid;
			warptable[i].offset = offset++;
			count--;
			if (count == 0)
				return;
		}
	}
	goto again;
}

static __device__ void
do_scheduler(unsigned tableid)
{
	while (!*(volatile BOOL *)&d_fkinfo->sched_done) {
		unsigned	eNum = threadIdx.x;

		while (eNum < d_numEntriesPerPool) {
			int	taskid = tableid * d_numEntriesPerPool + eNum + 2;
			tentry_t	*tentry = d_taskentries + taskid - 2;
			int	ready = *(volatile int *)&tentry->ready;

			if (ready > 1) {
				tentry_t	*tentry_prev = d_taskentries + tentry->ready - 2;

				if (tentry->ready != taskid && tentry_prev->ready != -1) {
					__threadfence();
					continue;
				}
				else {
					tentry->ready = -1;
					tentry_prev->ready = 1;
					tentry_prev->sched = 1;
				}
			}
			if (tentry->sched) {
				tentry->sched = 0;
				doneCtr[eNum] = tentry->skrun.n_tbs * tentry->skrun.n_mtbs_per_tb;
				assign_task(taskid, doneCtr[eNum]);
			}
			eNum += 32;
		}
	}
}

static __device__ void
do_executer(unsigned tableid)
{
	wentry_t	*wentry = &warptable[threadIdx.x / 32 - 1];

	while (!*(volatile BOOL *)&d_fkinfo->sched_done) {
		tentry_t	*tentry;

		while (!wentry->exec || !wentry->taskid) {
			if (d_fkinfo->sched_done)
				return;
			sleep_in_kernel();
		}
		tentry = d_taskentries + wentry->taskid - 2;
		run_sub_kernel(&tentry->skrun);
		if (get_laneid() == 0) {
			int	*ready_host = d_readytable + wentry->taskid - 2;
			int	eNum = (wentry->taskid - 2) % d_numEntriesPerPool;
			wentry->taskid = 0;
			wentry->exec = 0;
			if (atomicDec(doneCtr + eNum, MAX_ENTRIES_PER_POOL) == 1) {
				*ready_host = 0;
				tentry->ready = 0;
			}
		}
		SYNCWARP();
	}
}

extern "C" __global__ void
pagoda_master_kernel(void)
{
	unsigned	tableid = blockIdx.x * 2 + blockIdx.y;

	if (threadIdx.x < 32) {
		do_scheduler(tableid);
	}
	else {
		do_executer(tableid);
	}
}
