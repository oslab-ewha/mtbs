#include "tbs_sd.h"

#define SKRID_MY()	skrid_offs[d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB].skrid
#define OFFSET_MY()	skrid_offs[d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB].offset

static __device__ BOOL	*d_mtbs_done;

static __device__ volatile unsigned	cur_skr_idx;

typedef struct {
	skrid_t	skrid;
	unsigned short	offset;
} skrid_off_t;

static __shared__ skrid_off_t	skrid_offs[64];

__device__ static volatile int	in_scheduling;

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

static __device__ void
assign_my_task(void)
{
	while (!d_fkinfo->going_to_shutdown && !*(volatile BOOL *)&d_fkinfo->sched_done) {
		if (lock_scheduling() == 0) {
			skrun_t	*skr = &d_skruns[cur_skr_idx];
			skid_t	skid;
			skid = *(volatile skid_t *)(&skr->skid);
			if (skid != 0 && *(volatile unsigned *)&skr->n_mtbs_per_tb > 0) {
				SKRID_MY() = cur_skr_idx + 1;
				cur_skr_idx = (cur_skr_idx + 1) % d_fkinfo->n_queued_kernels;
				unlock_scheduling();
				return;
			}
			unlock_scheduling();
		}
		sleep_in_kernel();
	}
	SKRID_MY() = 0;
}

static __device__ skrid_t
get_skrid_gemtc(void)
{
	if (IS_LEADER_THREAD())
		assign_my_task();

	SYNCWARP();

	return SKRID_MY();
}

extern "C" __global__ void
func_macro_TB_gemtc(void)
{
	while (!*(volatile BOOL *)&d_fkinfo->going_to_shutdown) {
		skrid_t	skrid;
		skrun_t	*skr;

		skrid = get_skrid_gemtc();
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		for (unsigned i = 0; i < skr->n_mtbs_per_tb * skr->n_tbs; i++) {
			if (IS_LEADER_THREAD())
				OFFSET_MY() = i;
			SYNCWARP();

			run_sub_kernel(skr);
		}
		if (IS_LEADER_THREAD()) {
			d_mtbs_done[skrid - 1] = TRUE;
			skr->skid = 0;
		}
	}
}

static __device__ skrun_t *
get_skr_gemtc(void)
{
	return &d_skruns[SKRID_MY() - 1];
}

static __device__ unsigned short
get_offset_TB_gemtc(void)
{
	return OFFSET_MY();
}

extern "C" __global__ void
setup_sched_gemtc(skrun_t *skruns, BOOL *mtbs_done)
{
	int	i;

	d_skruns = skruns;
	d_mtbs_done = mtbs_done;

	for (i = 0; i < d_fkinfo->n_queued_kernels; i++) {
		skruns[i].skid = 0;
	}

	benchapi_funcs.get_skr = get_skr_gemtc;
	benchapi_funcs.get_offset_TB = get_offset_TB_gemtc;
}
