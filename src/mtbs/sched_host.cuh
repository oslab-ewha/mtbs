#include "tbs_sd.h"

#define mTB_INDEX_MY()		(blockIdx.x * dn_mtbs_per_sm + d_fkinfo->n_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)

#define EPOCH_MY()		mtb_epochs[mTB_INDEX_MY() - 1]

#define mTB_ALLOCOFF_TABLE_MY()	(mAOTs + dn_mtbs * EPOCH_MY())

#define SKRID_MY()		mTB_ALLOCOFF_TABLE_MY()[mTB_INDEX_MY() - 1].skrid
#define OFFSET_MY()		mTB_ALLOCOFF_TABLE_MY()[mTB_INDEX_MY() - 1].offset
#define BARID_MY()		mTB_ALLOCOFF_TABLE_MY()[mTB_INDEX_MY() - 1].barid

/* epoch directory for mTB allocation table */
static __device__ volatile mAO_t	*mAOTs;
static __device__ volatile unsigned char	*mtb_epochs;

static __device__ BOOL	*d_mtbs_done;
static __device__ unsigned	*d_mtbs_done_cnts;

static __shared__ unsigned short	offsets[64];

static __device__ unsigned	dn_mtbs;
static __device__ unsigned	dn_mtbs_per_sm;

static __device__ skrid_t
get_skrid_host(void)
{
	for (;;) {
		skrid_t	skrid;

		skrid = *(volatile unsigned short *)&SKRID_MY();
		if (skrid != 0 && *(volatile unsigned *)&d_skruns[skrid - 1].n_mtbs_per_tb > 0) {
			return skrid;
		}
		if (d_fkinfo->going_to_shutdown || *(volatile BOOL *)&d_fkinfo->sched_done)
			break;

		SYNCWARP();
	}

	return 0;
}

static __device__ void
advance_epoch_host(skrid_t skrid)
{
	if (IS_LEADER_THREAD()) {
		skrun_t	*skr = &d_skruns[skrid - 1];

		SKRID_MY() = 0;
		EPOCH_MY() = (EPOCH_MY() + 1) % EPOCH_MAX;
		if (atomicAdd(d_mtbs_done_cnts + skrid - 1, 1) == skr->n_mtbs_per_tb - 1) {
			d_mtbs_done[skrid - 1] = TRUE;
			d_mtbs_done_cnts[skrid - 1] = 0;
		}
	}
	SYNCWARP();
}

static __device__ skrun_t *
get_skr_host(void)
{
	skrid_t		skrid = SKRID_MY();

	return &d_skruns[skrid - 1];
}

static __device__ unsigned short
get_offset_TB_host(void)
{
	return OFFSET_MY();
}

static __device__ unsigned
get_barid_host(skrun_t *skr)
{
	return BARID_MY();
}

extern "C" __global__ void
func_macro_TB_host(void)
{
	while (!d_fkinfo->going_to_shutdown) {
		skrid_t	skrid;
		skrun_t	*skr;

		skrid = get_skrid_host();
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		for (unsigned i = 0; i < skr->n_tbs; i++) {
			SYNCWARP();

			run_sub_kernel(skr);
			if (IS_LEADER_THREAD())
				OFFSET_MY() = OFFSET_MY() + skr->n_mtbs_per_tb;
		}
		advance_epoch_host(skrid);
	}
}

extern "C" __global__ void
func_init_skrun_host(mAO_t *g_mAOTs, unsigned char *g_mtb_epochs, skrun_t *skruns, BOOL *mtbs_done)
{
	int	size;
	int	i;

	dn_mtbs_per_sm = d_fkinfo->n_mtbs_per_MTB * d_fkinfo->n_MTBs_per_sm;
	dn_mtbs = dn_mtbs_per_sm * d_fkinfo->n_sm_count;

	size = EPOCH_MAX * dn_mtbs;

	d_skruns = skruns;
	d_mtbs_done = mtbs_done;
	d_mtbs_done_cnts = (unsigned *)malloc(d_fkinfo->n_queued_kernels * sizeof(unsigned));

	for (i = 0; i < d_fkinfo->n_queued_kernels; i++) {
		skruns[i].skid = 0;
	}

	mAOTs = g_mAOTs;
	mtb_epochs = g_mtb_epochs;

	for (i = 0; i < size; i++) {
		mAOTs[i].skrid = 0;
		mAOTs[i].offset = 0;
	}

	for (i = 0; i < dn_mtbs; i++) {
		mtb_epochs[i] = 0;
	}

	benchapi_funcs.get_skr = get_skr_host;
	benchapi_funcs.get_offset_TB = get_offset_TB_host;
	benchapi_funcs.get_barid = get_barid_host;
}
