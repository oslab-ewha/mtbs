#include "tbs_sd.h"

#include "mAT.h"

__device__ static volatile int	in_scheduling;

/* number of scheduled mtbs per skr */
__device__ static volatile unsigned	*skr_n_tbs_sched;

__device__ static volatile unsigned	cur_skrid;

__device__ unsigned cu_get_tb_sm_rr(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);
__device__ unsigned cu_get_tb_sm_rrf(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);
__device__ unsigned cu_get_tb_sm_fca(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);
__device__ unsigned cu_get_tb_sm_rrm(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);

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
	while (!going_to_shutdown) {
		skrun_t	*skr = &d_skruns[cur_skrid];
		skid_t	skid;
		skid = *(volatile skid_t *)(&skr->skid);
		if (skid != 0) {
			skrid_t	skrid = cur_skrid + 1;

			if (SKR_N_TBS_SCHED(skrid) == skr->n_tbs) {
				skr->skid = 0;
				cur_skrid = (cur_skrid + 1) % dn_queued_kernels;
				continue;
			}
			return skrid;
		}
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

	switch (d_fkinfo->sched_id) {
	case 2:
		id_sm_sched = cu_get_tb_sm_rr(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 3:
		id_sm_sched = cu_get_tb_sm_rrf(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 4:
		id_sm_sched = cu_get_tb_sm_fca(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 5:
		id_sm_sched = cu_get_tb_sm_rrm(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	default:
		break;
	}

	if (id_sm_sched == 0)
		return FALSE;

	off_tb_base = SKR_N_TBS_SCHED(skrid) * skr->n_mtbs_per_tb;
	for (i = 0; i < skr->n_mtbs_per_tb; i++) {
		SKRID(id_sm_sched, idx_mtb_start + i) = skrid;
		mTB_OFFSET_TB(id_sm_sched, idx_mtb_start + i) = off_tb_base + i;
		mTB_SYNC(id_sm_sched, idx_mtb_start + i) = 0;
	}
	SKR_N_TBS_SCHED(skrid)++;
	return TRUE;
}

static __device__ void
run_schedule_in_kernel(void)
{
	if (lock_scheduling() < 0)
		return;

	if (going_to_shutdown) {
		unlock_scheduling();
		return;
	}

	assign_tb();

	unlock_scheduling();
}

__device__ unsigned
find_mtb_start(unsigned id_sm, unsigned idx_mtb_start, unsigned n_mtbs)
{
	int	i;

	for (i = idx_mtb_start; i <= d_fkinfo->n_max_mtbs_per_sm; i++) {
		if (SKRID(id_sm, i) == 0) {
			if (n_mtbs == 1)
				return i;
			if (i + n_mtbs - 1 <= d_fkinfo->n_max_mtbs_per_sm) {
				int	j;
				for (j = 1; j < n_mtbs; j++) {
					if (SKRID(id_sm, i + j) != 0)
						break;
				}
				if (j == n_mtbs)
					return i;
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

	for (i = 1; i <= d_fkinfo->n_max_mtbs_per_sm; i++) {
		if (SKRID(id_sm, i) != 0)
			count++;
	}
	return count;
}

__device__ skrun_t *
get_skr_dyn(void)
{
	unsigned	id_sm = get_smid() + 1;
	skrid_t		skrid = SKRID_MY(id_sm);

	return &d_skruns[skrid - 1];
}

__device__ unsigned
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

extern "C" __global__ void
setup_sched_dyn(void)
{
	int	i;

	run_schedule = run_schedule_in_kernel;

	skr_n_tbs_sched = (unsigned *)malloc(dn_queued_kernels * sizeof(unsigned));
	for (i = 0; i < dn_queued_kernels; i++) {
		skr_n_tbs_sched[i] = 0;
	}
}
