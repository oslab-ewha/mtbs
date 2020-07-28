#include "mtbs_cu.h"

#include <pthread.h>

#include "mAT.h"

static pthread_t	host_scheduler;
static BOOL	host_scheduler_done;

static unsigned		id_sm_last = 1;

static CUstream		strm_sched;

static unsigned short	*mATs_host;
static unsigned	char	*mtb_epochs_host;
static unsigned	char	*mtb_epochs_host_alloc;

static unsigned	mAT_uprange_start, mAT_uprange_end;

typedef struct {
	skrid_t		skrid;
	unsigned char	*epochs;
} sched_ctx_t;

static pthread_spinlock_t	lock;
static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;

#define NEXT_EPOCH(epoch)		(((epoch) + 1)  % EPOCH_MAX)
#define mTB_INDEX_HOST(id_sm, idx)	((id_sm - 1) * n_max_mtbs_per_sm + idx)
#define EPOCH_HOST(id_sm, idx)		mtb_epochs_host[mTB_INDEX_HOST(id_sm, idx) - 1]
#define EPOCH_HOST_ALLOC(id_sm, idx)	mtb_epochs_host_alloc[mTB_INDEX_HOST(id_sm, idx) - 1]
#define mTB_ALLOC_TABLE_EPOCH_HOST(epoch)	(mATs_host + n_max_mtbs * (epoch))
#define SKRID_EPOCH_HOST(epoch, id_sm, idx)	mTB_ALLOC_TABLE_EPOCH_HOST(epoch)[mTB_INDEX_HOST(id_sm, idx) - 1]

#define SET_ID_SM_NEXT(id_sm)	do { (id_sm) = (id_sm + 1) % n_sm_count; \
		if ((id_sm) == 0) (id_sm) = n_sm_count; } while (0)

extern unsigned short	*g_mATs;
extern unsigned char	*g_mtb_epochs;

static BOOL
find_mtbs_on_sm(unsigned id_sm, unsigned n_mtbs, unsigned char *epochs)
{
	unsigned	n_mtbs_cur = 0;
	int	i;

	for (i = 1; i <= n_max_mtbs_per_sm; i++) {
		int	epoch = EPOCH_HOST(id_sm, i);
		int	epoch_alloc = EPOCH_HOST_ALLOC(id_sm, i);

		/* Next epoch entry should be set to zero to proect overrun */
		if (NEXT_EPOCH(NEXT_EPOCH(epoch_alloc)) == epoch)
			continue;

		epochs[i - 1] = epoch_alloc;
		n_mtbs_cur++;
		if (n_mtbs_cur == n_mtbs)
			return TRUE;
	}
	return FALSE;
}

static void
apply_mAT_uprange(sched_ctx_t *pctx, unsigned char epoch, unsigned id_sm, unsigned idx)
{
	unsigned	up_idx;

	pthread_spin_lock(&lock);
	up_idx = n_max_mtbs * epoch + mTB_INDEX_HOST(id_sm, idx) - 1;

	if (up_idx < mAT_uprange_start)
		mAT_uprange_start = up_idx;
	else if (up_idx >= mAT_uprange_end)
		mAT_uprange_end = up_idx + 1;
	pthread_spin_unlock(&lock);
}

static void
set_mtbs_skrid(sched_ctx_t *pctx, unsigned id_sm, unsigned n_mtbs, unsigned char *epochs)
{
	unsigned	n_mtbs_cur = 0;
	unsigned	i;

	for (i = 1; i <= n_max_mtbs_per_sm; i++) {
		unsigned char	epoch = epochs[i - 1];

		if (epoch == EPOCH_MAX)
			continue;
		SKRID_EPOCH_HOST(epoch, id_sm, i) = pctx->skrid;
		SKRID_EPOCH_HOST(NEXT_EPOCH(epoch), id_sm, i) = 0;
		apply_mAT_uprange(pctx, epoch, id_sm, i);
		apply_mAT_uprange(pctx, NEXT_EPOCH(epoch), id_sm, i);
		EPOCH_HOST_ALLOC(id_sm, i) = NEXT_EPOCH(epoch);
		n_mtbs_cur++;
		if (n_mtbs_cur == n_mtbs)
			return;
	}
}

static BOOL
assign_tb_by_rr(sched_ctx_t *pctx, unsigned n_mtbs)
{
	unsigned	id_sm_start;
	int	id_sm_cur;

	pthread_mutex_lock(&mutex);

	id_sm_start = id_sm_last;
	id_sm_cur = id_sm_last;

	do {
		memset(pctx->epochs, EPOCH_MAX, n_max_mtbs_per_sm);
		if (find_mtbs_on_sm(id_sm_cur, n_mtbs, pctx->epochs)) {
			set_mtbs_skrid(pctx, id_sm_cur, n_mtbs, pctx->epochs);
			SET_ID_SM_NEXT(id_sm_cur);
			id_sm_last = id_sm_cur;

			pthread_mutex_unlock(&mutex);
			return TRUE;
		}
		SET_ID_SM_NEXT(id_sm_cur);
	}
	while (id_sm_start != id_sm_cur);

	pthread_mutex_unlock(&mutex);

	return FALSE;
}

static void
reload_epochs(void)
{
	cuMemcpyDtoHAsync(mtb_epochs_host, (CUdeviceptr)g_mtb_epochs, n_max_mtbs, strm_sched);
	cuStreamSynchronize(strm_sched);
}

static void
assign_tb_host(sched_ctx_t *pctx, unsigned n_mtbs_per_tb)
{
	do {
		if (assign_tb_by_rr(pctx, n_mtbs_per_tb))
			return;
		reload_epochs();
	} while (TRUE);
}

static void
init_sched_ctx(sched_ctx_t *pctx, skrid_t skrid)
{
	pctx->skrid = skrid;
	pctx->epochs = (unsigned char *)malloc(n_max_mtbs_per_sm);
}

static void
fini_sched_ctx(sched_ctx_t *pctx)
{
	free(pctx->epochs);
}

static void
schedule_mtbs(skrid_t skrid, unsigned n_tbs, unsigned n_mtbs_per_tb)
{
	sched_ctx_t	ctx;
	int	i;

	init_sched_ctx(&ctx, skrid);
	for (i = 0; i < n_tbs; i++) {
		assign_tb_host(&ctx, n_mtbs_per_tb);
	}
	fini_sched_ctx(&ctx);
}

static sk_t
submit_skrun_static(vstream_t vstream, skrun_t *skr)
{
	skrid_t	skrid;

	skrid = submit_skrun_mAT(skr);

	schedule_mtbs(skrid, skr->n_tbs, skr->n_mtbs_per_tb);

	return (sk_t)(long long)skrid;
}

static void
wait_skrun_static(sk_t sk, vstream_t vstream, int *pres)
{
	wait_skrun_mAT(sk, pres);
}

static void
update_mAT(CUstream strm)
{
	unsigned	len;
	unsigned	start, end;

	pthread_spin_lock(&lock);
	if (mAT_uprange_start == 0 && mAT_uprange_end == 0) {
		pthread_spin_unlock(&lock);
		return;
	}

	start = mAT_uprange_start;
	end = mAT_uprange_end;
	mAT_uprange_start = 0;
	mAT_uprange_end = 0;

	pthread_spin_unlock(&lock);

	len = end - start;

	cuMemcpyHtoDAsync((CUdeviceptr)(g_mATs + start), mATs_host + start, len * sizeof(unsigned short), strm);
	cuStreamSynchronize(strm);
}

static void *
host_schedfunc(void *arg)
{
	CUstream	strm;

	cuCtxSetCurrent(context);

	cuStreamCreate(&strm, CU_STREAM_NON_BLOCKING);

	while (!host_scheduler_done) {
		pthread_mutex_lock(&mutex);
		update_mAT(strm);
		pthread_mutex_unlock(&mutex);
		usleep(10);
	}

	cuStreamDestroy(strm);
	return NULL;
}

static void
init_skrun_static(void)
{
	int	i;

	init_skrun_mAT();

	cuStreamCreate(&strm_sched, CU_STREAM_NON_BLOCKING);

	mATs_host = (unsigned short *)malloc(EPOCH_MAX * n_max_mtbs * sizeof(unsigned short));
	for (i = 0; i < n_max_mtbs * EPOCH_MAX; i++) {
		mATs_host[i] = 0;
	}

	mtb_epochs_host = (unsigned char *)malloc(n_max_mtbs);
	mtb_epochs_host_alloc = (unsigned char *)malloc(n_max_mtbs);
	for (i = 0; i < n_max_mtbs; i++) {
		mtb_epochs_host[i] = 0;
		mtb_epochs_host_alloc[i] = 0;
	}

	pthread_spin_init(&lock, 0);

	pthread_create(&host_scheduler, NULL, host_schedfunc, NULL);
}

static void
fini_skrun_static(void)
{
	void	*retval;

	host_scheduler_done = TRUE;
	pthread_join(host_scheduler, &retval);

	fini_skrun_mAT();
}

sched_t	sched_sd_static = {
	"static",
	TBS_TYPE_SD_STATIC,
	"func_macro_TB_mAT",
	init_skrun_static,
	fini_skrun_static,
	submit_skrun_static,
	wait_skrun_static,
};
