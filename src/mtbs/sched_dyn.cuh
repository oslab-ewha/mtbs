#include "tbs_sd.h"

#define EPOCH_MAX	64

#define mTB_TOTAL_COUNT()      (dn_mtbs_per_sm * d_fkinfo->n_sm_count)
#define SKR_N_TBS_SCHED(skrid) skr_n_tbs_sched[skrid - 1]

#define mTB_INDEX(id_sm, idx)  ((id_sm - 1) * dn_mtbs_per_sm + idx)
#define mTB_INDEX_MY(id_sm)    ((id_sm - 1) * dn_mtbs_per_sm + d_fkinfo->n_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)

#define EPOCH(id_sm, idx)      mtb_epochs[mTB_INDEX(id_sm, idx) - 1]
#define EPOCH_MY(id_sm)                mtb_epochs[mTB_INDEX_MY(id_sm) - 1]

#define mTB_ALLOC_TABLE(id_sm, idx)    (mATs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_ALLOC_TABLE_MY(id_sm)      (mATs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))
#define SKRID(id_sm, idx)      mTB_ALLOC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define SKRID_MY(id_sm)                mTB_ALLOC_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define mTB_OFFSET_TABLE(id_sm, idx)   (mOTs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_OFFSET_TABLE_MY(id_sm)     (mOTs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))

#define mTB_OFFSET_TB(id_sm, idx)      mTB_OFFSET_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define mTB_OFFSET_TB_MY(id_sm)                mTB_OFFSET_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define mTB_SYNC_TABLE(id_sm, idx)    (mSTs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_SYNC(id_sm, idx)   mTB_SYNC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]

__device__ static volatile int	in_scheduling;

/* number of scheduled mtbs per skr */
__device__ static volatile unsigned	*skr_n_tbs_sched;

__device__ static volatile unsigned	cur_skr_idx;

__device__ unsigned cu_get_tb_sm_rr(unsigned n_mtbs, unsigned *pidx_mtb_start);

/* epoch directory for mTB allocation table */
__device__ volatile unsigned short	*mATs;
__device__ volatile unsigned char	*mtb_epochs;

/* offset in TB per mTB */
__device__ volatile unsigned short	*mOTs;
/* sync counter per mTB */
__device__ volatile unsigned short	*mSTs;

static __device__ BOOL	*d_mtbs_done;
static __device__ unsigned	*d_mtbs_done_cnts;

static __device__ unsigned	dn_mtbs_per_sm;

static __device__ int
lock_scheduling(void)
{
	if (threadIdx.x % N_THREADS_PER_mTB != 0)
		return -1;
	if (atomicCAS((int *)&in_scheduling, 0, 1) != 0)
		return -1;
	return 0;
}

static __device__ void
unlock_scheduling(void)
{
	in_scheduling = 0;
}

static __device__ skrid_t
get_sched_skrid(void)
{
	while (!d_fkinfo->going_to_shutdown) {
		skrun_t	*skr = &d_skruns[cur_skr_idx];
		skid_t	skid;
		skid = *(volatile skid_t *)(&skr->skid);
		if (skid != 0)
			return cur_skr_idx + 1;
		if (*(volatile BOOL *)&d_fkinfo->sched_done)
			return 0;
		sleep_in_kernel();
	}
	return 0;
}

static __device__ BOOL
assign_tb(void)
{
	skrun_t		*skr;
	unsigned	id_sm_sched;
	unsigned	idx_mtb_start;
	unsigned	off_tb_base;
	skrid_t		skrid;
	int		i;

	skrid = get_sched_skrid();
	if (skrid == 0)
		return FALSE;

	skr = &d_skruns[skrid - 1];

	id_sm_sched = cu_get_tb_sm_rr(skr->n_mtbs_per_tb, &idx_mtb_start);

	if (id_sm_sched == 0)
		return FALSE;

	off_tb_base = SKR_N_TBS_SCHED(skrid) * skr->n_mtbs_per_tb;
	for (i = 0; i < skr->n_mtbs_per_tb; i++) {
		SKRID(id_sm_sched, idx_mtb_start + i) = skrid;
		mTB_OFFSET_TB(id_sm_sched, idx_mtb_start + i) = off_tb_base + i;
		mTB_SYNC(id_sm_sched, idx_mtb_start + i) = 0;
	}
	SKR_N_TBS_SCHED(skrid)++;
	if (SKR_N_TBS_SCHED(skrid) == skr->n_tbs) {
		cur_skr_idx = (cur_skr_idx + 1) % d_fkinfo->n_queued_kernels;
		SKR_N_TBS_SCHED(skrid) = 0;
	}
	return TRUE;
}

static __device__ void
run_schedule_in_kernel(void)
{
	if (lock_scheduling() < 0)
		return;

	if (d_fkinfo->going_to_shutdown) {
		unlock_scheduling();
		return;
	}

	assign_tb();

	unlock_scheduling();
}

__device__ unsigned
find_mtb_start(unsigned id_sm, unsigned n_mtbs)
{
	int	i, j;
	unsigned	idx = 1;

	for (j = 0; j < d_fkinfo->n_MTBs_per_sm; j++) {
		for (i = 0; i < d_fkinfo->n_mtbs_per_MTB; i++, idx++) {
			if (SKRID(id_sm, idx) == 0) {
				if (n_mtbs == 1)
					return idx;
				if (((idx - 1) % d_fkinfo->n_mtbs_per_MTB) + n_mtbs <= d_fkinfo->n_mtbs_per_MTB) {
					int	k;
					for (k = 1; k < n_mtbs; k++) {
						if (SKRID(id_sm, idx + k) != 0)
							break;
					}
					if (k == n_mtbs) {
						return idx;
					}
				}
			}
		}
	}
	return 0;
}

__device__ unsigned
get_n_active_mtbs(unsigned id_sm)
{
	unsigned	count = 0;
	int	i;

	for (i = 1; i <= dn_mtbs_per_sm; i++) {
		if (SKRID(id_sm, i) != 0)
			count++;
	}
	return count;
}

static __device__ skrun_t *
get_skr_dyn(void)
{
	unsigned	id_sm = get_smid() + 1;
	skrid_t		skrid = SKRID_MY(id_sm);

	return &d_skruns[skrid - 1];
}

static __device__ unsigned short
get_offset_TB_dyn(void)
{
	unsigned	id_sm = get_smid() + 1;

	return mTB_OFFSET_TB_MY(id_sm);
}

__device__ void
sync_TB_threads_dyn(void)
{
	if (IS_LEADER_THREAD()) {
		skrun_t	*skr = get_skr_dyn();

		if (skr->n_mtbs_per_tb > 1) {
			unsigned	id_sm = get_smid() + 1;
			unsigned	offset = get_offset_TB_dyn();
			int		idx_sync = mTB_INDEX_MY(id_sm) - offset;

			atomicInc((unsigned *)&mTB_SYNC(id_sm, idx_sync), skr->n_mtbs_per_tb - 1);
			while (mTB_SYNC(id_sm, idx_sync) > 0) {
				printf("%d\n", mTB_SYNC(id_sm, idx_sync));
			}
		}
	}
	SYNCWARP();
}

static __device__ skrid_t
get_skrid_dyn(void)
{
	unsigned	id_sm;

	id_sm = get_smid() + 1;

	for (;;) {
		skrid_t	skrid;

		skrid = SKRID_MY(id_sm);
		if (skrid != 0)
			return skrid;
		if (d_fkinfo->going_to_shutdown || *(volatile BOOL *)&d_fkinfo->sched_done)
			break;

		if (IS_LEADER_THREAD())
			run_schedule_in_kernel();
		SYNCWARP();
	}

	return 0;
}

static __device__ void
advance_epoch_dyn(skrid_t skrid)
{
	unsigned	id_sm = get_smid() + 1;

	SYNCWARP();

	if (IS_LEADER_THREAD()) {
		skrun_t	*skr = &d_skruns[skrid - 1];

		SKRID_MY(id_sm) = 0;
		EPOCH_MY(id_sm) = (EPOCH_MY(id_sm) + 1) % EPOCH_MAX;
		if (atomicAdd(d_mtbs_done_cnts + skrid - 1, 1) == skr->n_mtbs_per_tb * skr->n_tbs - 1) {
			d_mtbs_done[skrid - 1] = TRUE;
			d_mtbs_done_cnts[skrid - 1] = 0;
			skr->skid = 0;
		}
	}
	SYNCWARP();
}

extern "C" __global__ void
func_macro_TB_dyn(void)
{
	while (!*(volatile BOOL *)&d_fkinfo->going_to_shutdown) {
		skrid_t	skrid;
		skrun_t	*skr;

		skrid = get_skrid_dyn();
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		run_sub_kernel(skr);

		advance_epoch_dyn(skrid);
	}
}

extern "C" __global__ void
setup_sched_dyn(unsigned short *g_mATs, unsigned char *g_mtb_epochs, skrun_t *skruns, BOOL *mtbs_done)
{
	int	size;
	int	i;

	dn_mtbs_per_sm = d_fkinfo->n_mtbs_per_MTB * d_fkinfo->n_MTBs_per_sm;
	size = EPOCH_MAX * mTB_TOTAL_COUNT();

	mOTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	mSTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	if (mOTs == NULL || mSTs == NULL) {
		printf("out of memory: offset or sync table cannot be allocated\n");
		d_fkinfo->going_to_shutdown = TRUE;
		return;
	}

	d_skruns = skruns;
	d_mtbs_done = mtbs_done;
	d_mtbs_done_cnts = (unsigned *)malloc(d_fkinfo->n_queued_kernels * sizeof(unsigned));

	for (i = 0; i < d_fkinfo->n_queued_kernels; i++) {
		skruns[i].skid = 0;
		d_mtbs_done_cnts[i] = 0;
	}

	mATs = g_mATs;
	mtb_epochs = g_mtb_epochs;

	for (i = 0; i < size; i++) {
		mATs[i] = 0;
		mOTs[i] = 0;
		mSTs[i] = 0;
	}

	for (i = 0; i < mTB_TOTAL_COUNT(); i++) {
		mtb_epochs[i] = 0;
	}

	skr_n_tbs_sched = (unsigned *)malloc(d_fkinfo->n_queued_kernels * sizeof(unsigned));
	for (i = 0; i < d_fkinfo->n_queued_kernels; i++) {
		skr_n_tbs_sched[i] = 0;
	}

	benchapi_funcs.get_skr = get_skr_dyn;
	benchapi_funcs.get_offset_TB = get_offset_TB_dyn;
	benchapi_funcs.sync_TB_threads = sync_TB_threads_dyn;
}
