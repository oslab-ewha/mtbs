#include "tbs_sd.h"

#define SKRID_MY()	skrid_offs[d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB].skrid
#define OFFSET_MY()	skrid_offs[d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB].offset

static __device__ BOOL	*d_mtbs_done;

static __device__ volatile unsigned	*cur_skr_idxs;

typedef struct {
	skrid_t	skrid;
	unsigned short	offset;
} skrid_off_t;

static __shared__ skrid_off_t	skrid_offs[64];

static __device__ volatile int	*in_schedulings;

static __device__ int
lock_scheduling(unsigned idx_sm)
{
	if (threadIdx.x % N_THREADS_PER_mTB != 0)
		return -1;
	if (atomicCAS((int *)&in_schedulings[idx_sm], 0, 1) != 0)
		return -1;
	return 0;
}

static __device__ void
unlock_scheduling(unsigned idx_sm)
{
	in_schedulings[idx_sm] = 0;
}

static __device__ void
assign_my_task(void)
{
	unsigned	idx_sm = get_smid();

	while (!d_fkinfo->going_to_shutdown && !*(volatile BOOL *)&d_fkinfo->sched_done) {
		if (lock_scheduling(idx_sm) == 0) {
			skrun_t	*skr = &d_skruns[idx_sm * d_fkinfo->n_queued_kernels + cur_skr_idxs[idx_sm]];
			skid_t	skid;
			skid = *(volatile skid_t *)(&skr->skid);
			if (skid != 0 && *(volatile unsigned *)&skr->n_mtbs_per_tb > 0) {
				SKRID_MY() = cur_skr_idxs[idx_sm] + 1;
				cur_skr_idxs[idx_sm] = (cur_skr_idxs[idx_sm] + 1) % d_fkinfo->n_queued_kernels;
				unlock_scheduling(idx_sm);
				return;
			}
			unlock_scheduling(idx_sm);
		}
		sleep_in_kernel();
	}
	SKRID_MY() = 0;
}

static __device__ skrid_t
get_skrid_gemtcP(void)
{
	if (IS_LEADER_THREAD())
		assign_my_task();

	SYNCWARP();

	return SKRID_MY();
}

extern "C" __global__ void
func_macro_TB_gemtcP(void)
{
	while (!*(volatile BOOL *)&d_fkinfo->going_to_shutdown) {
		skrid_t	skrid;
		skrun_t	*skr;
		unsigned short	id_sm = (unsigned short)get_smid();

		skrid = get_skrid_gemtcP();
		if (skrid == 0)
			return;

		skr = &d_skruns[id_sm * d_fkinfo->n_queued_kernels + skrid - 1];
		for (unsigned i = 0; i < skr->n_mtbs_per_tb * skr->n_tbs; i++) {
			if (IS_LEADER_THREAD())
				OFFSET_MY() = i;
			SYNCWARP();

			run_sub_kernel(skr);
		}
		d_mtbs_done[id_sm * d_fkinfo->n_queued_kernels + skrid - 1] = TRUE;
		skr->skid = 0;
	}
}

static __device__ skrun_t *
get_skr_gemtcP(void)
{
	return &d_skruns[get_smid() * d_fkinfo->n_queued_kernels + SKRID_MY() - 1];
}

static __device__ unsigned short
get_offset_TB_gemtcP(void)
{
	return OFFSET_MY();
}

extern "C" __global__ void
setup_sched_gemtcP(skrun_t *skruns, BOOL *mtbs_done)
{
	int	i;

	d_skruns = skruns;
	d_mtbs_done = mtbs_done;
	for (i = 0; i < d_fkinfo->n_sm_count * d_fkinfo->n_queued_kernels; i++) {
		skruns[i].skid = 0;
	}

	cur_skr_idxs = (unsigned *)malloc(sizeof(unsigned) * d_fkinfo->n_sm_count);
	in_schedulings = (volatile int *)malloc(sizeof(unsigned) * d_fkinfo->n_sm_count);
	for (i = 0; i < d_fkinfo->n_sm_count; i++) {
		cur_skr_idxs[i] = 0;
		in_schedulings[i] = 0;
	}

	benchapi_funcs.get_skr = get_skr_gemtcP;
	benchapi_funcs.get_offset_TB = get_offset_TB_gemtcP;
}
