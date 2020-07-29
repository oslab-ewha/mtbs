/* epoch directory for mTB allocation table */
__device__ volatile unsigned short	*mATs;
__device__ volatile unsigned char	*mtb_epochs;

/* offset in TB per mTB */
__device__ volatile unsigned short	*mOTs;
/* sync counter per mTB */
__device__ volatile unsigned short	*mSTs;

static __device__ BOOL	*d_mtbs_done;
static __device__ unsigned	*d_mtbs_done_cnts;

void __device__ (*run_schedule)(void);

static __device__ skrid_t
get_skrid_mAT(void)
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

		if (run_schedule && IS_LEADER_THREAD())
			run_schedule();
		SYNCWARP();
	}

	return 0;
}

static __device__ void
advance_epoch_mAT(skrid_t skrid)
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
func_macro_TB_mAT(void)
{
	while (!going_to_shutdown) {
		skrid_t	skrid;
		skrun_t	*skr;

		skrid = get_skrid_mAT();
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		run_sub_kernel(skr);

		advance_epoch_mAT(skrid);
	}
}

extern "C" __global__ void
func_init_skrun_mAT(unsigned short *g_mATs, unsigned char *g_mtb_epochs,
		    unsigned n_queued_kernels, skrun_t *skruns, BOOL *mtbs_done)
{
	int	size;
	int	i;

	size = EPOCH_MAX * mTB_TOTAL_COUNT();

	mOTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	mSTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	if (mOTs == NULL || mSTs == NULL) {
		printf("out of memory: offset or sync table cannot be allocated\n");
		going_to_shutdown = TRUE;
		return;
	}

	dn_queued_kernels = n_queued_kernels;
	d_skruns = skruns;
	d_mtbs_done = mtbs_done;
	d_mtbs_done_cnts = (unsigned *)malloc(n_queued_kernels * sizeof(unsigned));

	for (i = 0; i < dn_queued_kernels; i++) {
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
}
