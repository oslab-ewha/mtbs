#include "tbs_sd.h"

#define mTB_TOTAL_COUNT()	(d_fkinfo->n_max_mtbs_per_sm * d_fkinfo->n_sm_count)

#define mTB_INDEX_MY(id_sm)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)

#define EPOCH_MY(id_sm)		mtb_epochs[mTB_INDEX_MY(id_sm) - 1]

#define mTB_ALLOCOFF_TABLE_MY(id_sm)	(mAOTs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))

#define SKRID_MY(id_sm)		mTB_ALLOCOFF_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1].skrid
#define mTB_OFFSET_TB_MY(id_sm)		mTB_ALLOCOFF_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1].offset

/* epoch directory for mTB allocation table */
static __device__ volatile mAO_t	*mAOTs;
static __device__ volatile unsigned char	*mtb_epochs;

/* sync counter per mTB */
static __device__ volatile unsigned short	*mSTs;

static __device__ unsigned	*d_mtbs_done_cnts;

static __device__ skrid_t
get_skrid_static(void)
{
	unsigned	id_sm;

	id_sm = get_smid() + 1;

	for (;;) {
		skrid_t	skrid;

		skrid = SKRID_MY(id_sm);
		if (skrid != 0)
			return skrid;
		if (going_to_shutdown || *(volatile BOOL *)&d_fkinfo->sched_done)
			break;

		SYNCWARP();
	}

	return 0;
}

static __device__ void
advance_epoch_static(skrid_t skrid)
{
	unsigned	id_sm = get_smid() + 1;

	SYNCWARP();

	if (IS_LEADER_THREAD()) {
		SKRID_MY(id_sm) = 0;
		EPOCH_MY(id_sm) = (EPOCH_MY(id_sm) + 1) % EPOCH_MAX;
		atomicAdd(d_mtbs_done_cnts + skrid - 1, 1);
	}
	SYNCWARP();
}

static __device__ skrun_t *
get_skr_static(void)
{
	unsigned	id_sm = get_smid() + 1;
	skrid_t		skrid = SKRID_MY(id_sm);

	return &d_skruns[skrid - 1];
}

static __device__ unsigned short
get_offset_TB_static(void)
{
	unsigned	id_sm = get_smid() + 1;

	return mTB_OFFSET_TB_MY(id_sm);
}

extern "C" __global__ void
func_macro_TB_static(void)
{
	while (!going_to_shutdown) {
		skrid_t	skrid;
		skrun_t	*skr;

		skrid = get_skrid_static();
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		run_sub_kernel(skr);

		advance_epoch_static(skrid);
	}
}

extern "C" __global__ void
func_init_skrun_static(mAO_t *g_mAOTs, unsigned char *g_mtb_epochs,
		       unsigned n_queued_kernels, skrun_t *skruns, unsigned *mtbs_done_cnts)
{
	int	size;
	int	i;

	size = EPOCH_MAX * mTB_TOTAL_COUNT();

	mSTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	if (mSTs == NULL) {
		printf("out of memory: sync table cannot be allocated\n");
		going_to_shutdown = TRUE;
		return;
	}

	dn_queued_kernels = n_queued_kernels;
	d_skruns = skruns;
	d_mtbs_done_cnts = mtbs_done_cnts;
	for (i = 0; i < dn_queued_kernels; i++) {
		skruns[i].skid = 0;
		mtbs_done_cnts[i] = 0;
	}

	mAOTs = g_mAOTs;
	mtb_epochs = g_mtb_epochs;

	for (i = 0; i < size; i++) {
		mAOTs[i].skrid = 0;
		mAOTs[i].offset = 0;
		mSTs[i] = 0;
	}

	for (i = 0; i < mTB_TOTAL_COUNT(); i++) {
		mtb_epochs[i] = 0;
	}

	benchapi_funcs.get_skr = get_skr_static;
	benchapi_funcs.get_offset_TB = get_offset_TB_static;
}
