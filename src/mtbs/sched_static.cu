#include "mtbs_cu.h"

#include <pthread.h>

#define EPOCH_MAX		64

typedef struct {
	unsigned short	skrid;
	unsigned short	offset;
} mAO_t;

static pthread_t	host_scheduler;
static BOOL	host_scheduler_done;

static unsigned		id_sm_last = 1;

static CUstream		strm_static;

static mAO_t	*mAOTs_host;
static unsigned	char	*mtb_epochs_host;
static unsigned	char	*mtb_epochs_host_alloc;

static unsigned	mAT_uprange_start, mAT_uprange_end;

typedef struct {
	skrid_t		skrid;
	unsigned char	*epochs;
} sched_ctx_t;

static pthread_spinlock_t	lock;
static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	cond = PTHREAD_COND_INITIALIZER;

#define NEXT_EPOCH(epoch)		(((epoch) + 1)  % EPOCH_MAX)
#define mTB_INDEX_HOST(id_sm, idx)	((id_sm - 1) * n_max_mtbs_per_sm + idx)
#define EPOCH_HOST(id_sm, idx)		mtb_epochs_host[mTB_INDEX_HOST(id_sm, idx) - 1]
#define EPOCH_HOST_ALLOC(id_sm, idx)	mtb_epochs_host_alloc[mTB_INDEX_HOST(id_sm, idx) - 1]
#define mTB_ALLOCOFF_TABLE_EPOCH_HOST(epoch)	(mAOTs_host + n_max_mtbs * (epoch))
#define mAO_EPOCH_HOST(epoch, id_sm, idx)	(mTB_ALLOCOFF_TABLE_EPOCH_HOST(epoch) + mTB_INDEX_HOST(id_sm, idx) - 1)

#define SET_ID_SM_NEXT(id_sm)	do { (id_sm) = (id_sm + 1) % n_sm_count; \
		if ((id_sm) == 0) (id_sm) = n_sm_count; } while (0)

static skrun_t	*g_skruns;
static BOOL	*g_mtbs_done;

static BOOL	*skrun_dones;

static unsigned	skrid_done_min;
static unsigned	cur_skrid_host;

static BOOL	checker_done;
static pthread_t	checker;

static mAO_t	*g_mAOTs;
extern unsigned char	*g_mtb_epochs;

#include "sched_static.cuh"

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
set_mtbs_skrid(sched_ctx_t *pctx, unsigned id_sm, unsigned short offbase, unsigned n_mtbs, unsigned char *epochs)
{
	unsigned	n_mtbs_cur = 0;
	unsigned short	off = 0;
	unsigned	i;

	for (i = 1; i <= n_max_mtbs_per_sm; i++) {
		unsigned char	epoch = epochs[i - 1];
		mAO_t	*mAO, *mAO_next;

		if (epoch == EPOCH_MAX)
			continue;

		mAO = mAO_EPOCH_HOST(epoch, id_sm, i);
		mAO->skrid = pctx->skrid;
		mAO->offset = offbase + off;
		off++;
		mAO_next = mAO_EPOCH_HOST(NEXT_EPOCH(epoch), id_sm, i);
		mAO_next->skrid = 0;

		apply_mAT_uprange(pctx, epoch, id_sm, i);
		apply_mAT_uprange(pctx, NEXT_EPOCH(epoch), id_sm, i);
		EPOCH_HOST_ALLOC(id_sm, i) = NEXT_EPOCH(epoch);
		n_mtbs_cur++;
		if (n_mtbs_cur == n_mtbs)
			return;
	}
}

static BOOL
assign_tb_by_rr(sched_ctx_t *pctx, unsigned short offbase, unsigned n_mtbs)
{
	unsigned	id_sm_start;
	int	id_sm_cur;

	pthread_mutex_lock(&mutex);

	id_sm_start = id_sm_last;
	id_sm_cur = id_sm_last;

	do {
		memset(pctx->epochs, EPOCH_MAX, n_max_mtbs_per_sm);
		if (find_mtbs_on_sm(id_sm_cur, n_mtbs, pctx->epochs)) {
			set_mtbs_skrid(pctx, id_sm_cur, offbase, n_mtbs, pctx->epochs);
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
	cuMemcpyDtoHAsync(mtb_epochs_host, (CUdeviceptr)g_mtb_epochs, n_max_mtbs, strm_static);
	cuStreamSynchronize(strm_static);
}

static void
assign_tb_host(sched_ctx_t *pctx, unsigned short offbase, unsigned n_mtbs_per_tb)
{
	do {
		if (assign_tb_by_rr(pctx, offbase, n_mtbs_per_tb))
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
	unsigned short	offbase = 0;
	int	i;

	init_sched_ctx(&ctx, skrid);
	for (i = 0; i < n_tbs; i++) {
		assign_tb_host(&ctx, offbase, n_mtbs_per_tb);
		offbase += n_mtbs_per_tb;
	}
	fini_sched_ctx(&ctx);
}

static sk_t
submit_skrun_static(vstream_t vstream, skrun_t *skr)
{
	skrid_t	skrid;

	pthread_mutex_lock(&mutex);

	while (skrid_done_min == (cur_skrid_host + 1) % n_queued_kernels) {
		/* full */
		pthread_cond_wait(&cond, &mutex);
	}

	skrid = cur_skrid_host + 1;
	skrun_dones[skrid - 1] = FALSE;
	cuMemcpyHtoDAsync((CUdeviceptr)(g_skruns + cur_skrid_host), skr, sizeof(skrun_t), strm_static);
	/* No synchronization needed */

	cur_skrid_host = (cur_skrid_host + 1) % n_queued_kernels;

	pthread_mutex_unlock(&mutex);

	schedule_mtbs(skrid, skr->n_tbs, skr->n_mtbs_per_tb);

	return (sk_t)(long long)skrid;
}

static void
wait_skrun_static(sk_t sk, vstream_t vstream, int *pres)
{
	skrun_t	*skr;
	skrid_t	skrid = (skrid_t)(long long)sk;

	pthread_mutex_lock(&mutex);

	while (!checker_done && !skrun_dones[skrid - 1])
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);

	skr = g_skruns + (skrid - 1);
	cuMemcpyDtoHAsync(pres, (CUdeviceptr)&skr->res, sizeof(int), strm_static);
	cuStreamSynchronize(strm_static);
}

static void
update_mAOT(CUstream strm)
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

	cuMemcpyHtoDAsync((CUdeviceptr)(g_mAOTs + start), mAOTs_host + start, len * sizeof(mAO_t), strm);
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
		update_mAOT(strm);
		pthread_mutex_unlock(&mutex);
		usleep(10);
	}

	cuStreamDestroy(strm);
	return NULL;
}

static void
notify_done_skruns(unsigned n_checks)
{
	unsigned	min_new = skrid_done_min;
	BOOL		notify = FALSE;
	unsigned	i, idx;

	idx = skrid_done_min;
	for (i = 0; i < n_checks; i++) {
		if (!skrun_dones[idx]) {
			if (g_mtbs_done[idx]) {
				notify = TRUE;
				skrun_dones[idx] = TRUE;
				g_mtbs_done[idx] = FALSE;
			}
		}
		if (skrun_dones[idx]) {
			if (min_new == idx) {
				min_new = (min_new + 1) % n_queued_kernels;
				notify = TRUE;
			}
		}
		idx = (idx + 1) % n_queued_kernels;
	}
	skrid_done_min = min_new;
	if (notify)
		pthread_cond_broadcast(&cond);
}

static void *
skruns_checkfunc(void *arg)
{
	while (!checker_done) {
		unsigned	n_checks = (cur_skrid_host + n_queued_kernels - skrid_done_min) % n_queued_kernels;
		pthread_mutex_lock(&mutex);

		if (n_checks > 0) {
			notify_done_skruns(n_checks);
		}

		pthread_mutex_unlock(&mutex);
		usleep(100);
	}

	return NULL;
}

static void
init_skrun_static(void)
{
	void	*params[5];
	int	i;

	cuStreamCreate(&strm_static, CU_STREAM_NON_BLOCKING);

	g_skruns = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t) * n_queued_kernels);
	cuMemAllocHost((void **)&g_mtbs_done, sizeof(BOOL) * n_queued_kernels);

	for (i = 0; i < n_queued_kernels; i++)
		g_mtbs_done[i] = FALSE;

	skrun_dones = (BOOL *)calloc(n_queued_kernels, sizeof(BOOL));

	pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	g_mAOTs = (mAO_t *)mtbs_cudaMalloc(EPOCH_MAX * n_max_mtbs * sizeof(mAO_t));
	g_mtb_epochs = (unsigned char *)mtbs_cudaMalloc(n_max_mtbs);

	params[0] = &g_mAOTs;
	params[1] = &g_mtb_epochs;
	params[2] = &n_queued_kernels;
	params[3] = &g_skruns;
	params[4] = &g_mtbs_done;
	if (!invoke_kernel_func("func_init_skrun_static", params)) {
		exit(12);
	}

	mAOTs_host = (mAO_t *)malloc(EPOCH_MAX * n_max_mtbs * sizeof(mAO_t));
	for (i = 0; i < n_max_mtbs * EPOCH_MAX; i++) {
		mAOTs_host[i].skrid = 0;
		mAOTs_host[i].offset = 0;
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

	checker_done = TRUE;
	pthread_join(checker, &retval);
	mtbs_cudaFree(g_skruns);

	mtbs_cudaFree(g_mAOTs);
	mtbs_cudaFree(g_mtb_epochs);
}

sched_t	sched_sd_static = {
	"static",
	TBS_TYPE_SD_STATIC,
	"func_macro_TB_static",
	init_skrun_static,
	fini_skrun_static,
	submit_skrun_static,
	wait_skrun_static,
};
