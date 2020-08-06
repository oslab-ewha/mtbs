#include "tbs_sd.h"

#define mTB_TOTAL_COUNT()	(d_fkinfo->n_max_mtbs_per_sm * d_fkinfo->n_sm_count)

#define mTB_INDEX_MY(id_sm)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)

#define EPOCH_MY(id_sm)		mtb_epochs[mTB_INDEX_MY(id_sm) - 1]

#define mTB_ALLOC_TABLE_MY(id_sm)	(mATs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))

#define SKRID_MY(id_sm)		mTB_ALLOC_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]
#define OFFSET_MY()	offsets[d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + threadIdx.x / N_THREADS_PER_mTB]

/* epoch directory for mTB allocation table */
static __device__ volatile unsigned short	*mATs;
static __device__ volatile unsigned char	*mtb_epochs;

/* sync counter per mTB */
static __device__ volatile unsigned short	*mSTs;

static __device__ BOOL	*d_mtbs_done;
static __device__ unsigned	*d_mtbs_done_cnts;

static __shared__ unsigned short	offsets[64];

static __device__ skrid_t
get_skrid_host(void)
{
	unsigned	id_sm;

	id_sm = get_smid() + 1;

	for (;;) {
		skrid_t	skrid;

		skrid = *(volatile unsigned short *)&SKRID_MY(id_sm);
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
		unsigned	id_sm = get_smid() + 1;
		skrun_t	*skr = &d_skruns[skrid - 1];

		SKRID_MY(id_sm) = 0;
		EPOCH_MY(id_sm) = (EPOCH_MY(id_sm) + 1) % EPOCH_MAX;
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
	unsigned	id_sm = get_smid() + 1;
	skrid_t		skrid = SKRID_MY(id_sm);

	return &d_skruns[skrid - 1];
}

static __device__ unsigned short
get_offset_TB_host(void)
{
	return OFFSET_MY();
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
			if (IS_LEADER_THREAD())
				OFFSET_MY() = i;
			SYNCWARP();

			run_sub_kernel(skr);
		}
		advance_epoch_host(skrid);
	}
}

extern "C" __global__ void
func_init_skrun_host(unsigned short *g_mATs, unsigned char *g_mtb_epochs, skrun_t *skruns, BOOL *mtbs_done)
{
	int	size;
	int	i;

	size = EPOCH_MAX * mTB_TOTAL_COUNT();

	mSTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	if (mSTs == NULL) {
		printf("out of memory: sync table cannot be allocated\n");
		d_fkinfo->going_to_shutdown = TRUE;
		return;
	}

	d_skruns = skruns;
	d_mtbs_done = mtbs_done;
	d_mtbs_done_cnts = (unsigned *)malloc(d_fkinfo->n_queued_kernels * sizeof(unsigned));

	for (i = 0; i < d_fkinfo->n_queued_kernels; i++) {
		skruns[i].skid = 0;
	}

	mATs = g_mATs;
	mtb_epochs = g_mtb_epochs;

	for (i = 0; i < size; i++) {
		mATs[i] = 0;
		mSTs[i] = 0;
	}

	for (i = 0; i < mTB_TOTAL_COUNT(); i++) {
		mtb_epochs[i] = 0;
	}

	benchapi_funcs.get_skr = get_skr_host;
	benchapi_funcs.get_offset_TB = get_offset_TB_host;
}
