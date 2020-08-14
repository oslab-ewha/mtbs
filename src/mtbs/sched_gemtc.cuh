#include "tbs_sd.h"

#define SKRID_MY()	skrid_offs[d_fkinfo->n_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB].skrid
#define OFFSET_MY()	skrid_offs[d_fkinfo->n_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB].offset
#define BARID_MY()	skrid_offs[d_fkinfo->n_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB].barid

static __device__ BOOL	*d_mtbs_done;
static __device__ unsigned	*d_mtbs_done_cnts;

static __device__ volatile unsigned	cur_skr_idx;
static __device__ volatile unsigned	cur_blockIdxX;
static __device__ volatile unsigned	cur_blockIdxY;
static __device__ volatile unsigned	mtbs_alloced;
static __device__ volatile unsigned char	barid_alloced;

typedef struct {
	skrid_t	skrid;
	unsigned short	offset;
	unsigned char	barid;
} skrid_off_t;

static __shared__ skrid_off_t	skrid_offs[64];
static __shared__ unsigned char	bar_used;

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
	atomicExch((int *)&in_scheduling, 0);
}

static __device__ unsigned char
get_barid(void)
{
	unsigned	i;

	for (i = 0; i < 16; i++) {
		if (!(bar_used & (1 << i))) {
			bar_used |= (1 << i);
			break;
		}
	}
	return i;
}

static __device__ void
put_barid(unsigned char barid)
{
	bar_used &= ~(1 << barid);
}

static __device__ void
assign_my_task(void)
{
	while (!d_fkinfo->going_to_shutdown && !*(volatile BOOL *)&d_fkinfo->sched_done) {
		if (lock_scheduling() == 0) {
			skrun_t	*skr = &d_skruns[cur_skr_idx];
			skid_t	skid;
			skid = *(volatile skid_t *)(&skr->skid);
			if (skid != 0) {
				unsigned	n_mtbs_per_tb = *(volatile unsigned *)&skr->n_mtbs_per_tb;
				if (n_mtbs_per_tb > 0 && (mtbs_alloced == 0 || (cur_blockIdxX == blockIdx.x && cur_blockIdxY == blockIdx.y))) {
					if (n_mtbs_per_tb > 1) {
						if (mtbs_alloced == 0) {
							barid_alloced = get_barid();
							cur_blockIdxX = blockIdx.x;
							cur_blockIdxY = blockIdx.y;
						}
						BARID_MY() = barid_alloced;
					}
					SKRID_MY() = cur_skr_idx + 1;
					OFFSET_MY() = mtbs_alloced;
					mtbs_alloced++;
					if (mtbs_alloced == n_mtbs_per_tb) {
						cur_skr_idx = (cur_skr_idx + 1) % d_fkinfo->n_queued_kernels;
						mtbs_alloced = 0;
					}
					unlock_scheduling();
					return;
				}
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
		for (unsigned i = 0; i < skr->n_tbs; i++) {
			run_sub_kernel(skr);
			if (IS_LEADER_THREAD())
				OFFSET_MY() = OFFSET_MY() + skr->n_mtbs_per_tb;
			SYNCWARP();
		}
		if (IS_LEADER_THREAD()) {
			if (atomicAdd(d_mtbs_done_cnts + skrid - 1, 1) == skr->n_mtbs_per_tb - 1) {
				while (lock_scheduling() != 0);
				put_barid(BARID_MY());
				unlock_scheduling();

				d_mtbs_done[skrid - 1] = TRUE;
				d_mtbs_done_cnts[skrid - 1] = 0;
				skr->skid = 0;
			}
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

static __device__ unsigned
get_barid_gemtc(skrun_t *skr)
{
	return (unsigned)BARID_MY();
}

extern "C" __global__ void
setup_sched_gemtc(skrun_t *skruns, BOOL *mtbs_done)
{
	int	i;

	d_skruns = skruns;
	d_mtbs_done = mtbs_done;
	d_mtbs_done_cnts = (unsigned *)malloc(d_fkinfo->n_queued_kernels * sizeof(unsigned));
	for (i = 0; i < d_fkinfo->n_queued_kernels; i++) {
		skruns[i].skid = 0;
		d_mtbs_done_cnts[i] = 0;
	}

	benchapi_funcs.get_skr = get_skr_gemtc;
	benchapi_funcs.get_offset_TB = get_offset_TB_gemtc;
	benchapi_funcs.get_barid = get_barid_gemtc;
}
