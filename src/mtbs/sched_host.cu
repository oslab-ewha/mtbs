#include "mtbs_cu.h"

#include <pthread.h>
#include "ecm_list.h"

#define EPOCH_MAX	64

typedef struct {
	unsigned short	skrid;
	unsigned short	offset;
} mAO_t;

static pthread_t	host_scheduler;
static BOOL	host_scheduler_done;

static CUstream		strm_host;

static unsigned	char	*mtb_epochs_host;
static unsigned	char	*mtb_epochs_host_alloc;

typedef struct {
	unsigned	start, len;
	struct list_head	list;
} uprange_t;

typedef struct {
	unsigned	n_skruns;
	skrid_t		skrid_start;
	skrun_t		*skruns;
	mAO_t		*mAOTs_host;
	struct list_head	upranges;
} htod_copyinfo_t;

#define COPYIDX_OTHER()	((copyidx + 1) % 2)

static unsigned		copyidx;
static htod_copyinfo_t	copyinfos[2];

typedef struct {
	skrid_t		skrid;
	unsigned char	*epochs;
} sched_ctx_t;

static pthread_spinlock_t	lock;
static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	cond = PTHREAD_COND_INITIALIZER;

#define NEXT_EPOCH(epoch)		(((epoch) + 1)  % EPOCH_MAX)
#define mTB_INDEX_HOST(idx_sm, idx_MTB, idx)	((idx_sm) * n_max_mtbs_per_sm + (idx_MTB) * n_mtbs_per_MTB + idx)
#define EPOCH_HOST(idx_sm, idx_MTB, idx)		mtb_epochs_host[mTB_INDEX_HOST(idx_sm, idx_MTB, idx) - 1]
#define EPOCH_HOST_ALLOC(idx_sm, idx_MTB, idx)		mtb_epochs_host_alloc[mTB_INDEX_HOST(idx_sm, idx_MTB, idx) - 1]
#define mTB_ALLOCOFF_TABLE_EPOCH_HOST(cinfo, epoch)	((cinfo)->mAOTs_host + n_max_mtbs * (epoch))
#define mAO_EPOCH_HOST(cinfo, epoch, idx_sm, idx_MTB, idx)	(mTB_ALLOCOFF_TABLE_EPOCH_HOST(cinfo, epoch) + mTB_INDEX_HOST(idx_sm, idx_MTB, idx) - 1)

static skrun_t	*g_skruns;
static BOOL	*g_mtbs_done;

static BOOL	*skrun_dones;

static unsigned	skrid_done_min;
static unsigned	cur_skrid_host;

static BOOL	checker_done;
static pthread_t	checker;

static mAO_t		*g_mAOTs;
extern unsigned char	*g_mtb_epochs;

#include "sched_host.cuh"

typedef struct {
	BOOL	locked;
	unsigned	idxs_last[8];	/* 8 should be larger than n_MTBs_per_sm */
} sched_sm_t;

static sched_sm_t	*sched_sms;
static unsigned	idx_ssm_last;

static void
lock_sm(unsigned *pidx_sm, unsigned *pidx_MTB)
{
	unsigned	idx_ssm_cur = 0, idx_ssm_start;
	unsigned	n_MTBs = n_sm_count * n_MTBs_per_sm;

	pthread_spin_lock(&lock);

	idx_ssm_start = idx_ssm_last;

	while (TRUE) {
		if (!sched_sms[idx_ssm_last].locked) {
			sched_sms[idx_ssm_last].locked = TRUE;
			idx_ssm_cur = idx_ssm_last;
			idx_ssm_last = (idx_ssm_last + 1) % n_MTBs;
			break;
		}
		idx_ssm_last = (idx_ssm_last + 1) % n_MTBs;
		if (idx_ssm_last == idx_ssm_start) {
			pthread_spin_unlock(&lock);
			usleep(1);
			pthread_spin_lock(&lock);
		}
	}
	pthread_spin_unlock(&lock);
	*pidx_sm = idx_ssm_cur % n_sm_count;
	*pidx_MTB = idx_ssm_cur / n_sm_count;
}

static void
unlock_sm(unsigned idx_sm, unsigned idx_MTB)
{
	pthread_spin_lock(&lock);
	sched_sms[idx_MTB * n_sm_count + idx_sm].locked = FALSE;
	pthread_spin_unlock(&lock);
}

static BOOL
find_mtbs_on_sm(unsigned idx_sm, unsigned idx_MTB, unsigned n_mtbs, unsigned char *epochs)
{
	sched_sm_t	*ssm = &sched_sms[idx_sm];
	unsigned	n_mtbs_cur = 0;
	unsigned	idx, idx_start;

	idx = idx_start = ssm->idxs_last[idx_MTB];

	do {
		int	epoch = EPOCH_HOST(idx_sm, idx_MTB, idx + 1);
		int	epoch_alloc = EPOCH_HOST_ALLOC(idx_sm, idx_MTB, idx + 1);

		/* Next epoch entry should be set to zero to proect overrun */
		if (NEXT_EPOCH(NEXT_EPOCH(epoch_alloc)) == epoch) {
			idx = (idx + 1) % n_mtbs_per_MTB;
			continue;
		}

		epochs[idx] = epoch_alloc;
		n_mtbs_cur++;
		idx = (idx + 1) % n_mtbs_per_MTB;

		if (n_mtbs_cur == n_mtbs) {
			ssm->idxs_last[idx_MTB] = idx;
			return TRUE;
		}
	}
	while (idx != idx_start);
	return FALSE;
}

static uprange_t *
create_uprange(unsigned up_idx)
{
	uprange_t	*ur;

	ur = (uprange_t *)malloc(sizeof(uprange_t));
	ur->start = up_idx;
	ur->len = 1;
	return ur;
}

static BOOL
get_sibling_upranges(htod_copyinfo_t *cinfo, unsigned up_idx, uprange_t **pprev, uprange_t **pnext)
{
	uprange_t	*prev = NULL;
	struct list_head	*lp;

	list_for_each (lp, &cinfo->upranges) {
		uprange_t	*ur = list_entry(lp, uprange_t, list);

		if (prev && prev->start <= up_idx && up_idx < prev->start + prev->len)
			return FALSE;
		if (up_idx == ur->start)
			return FALSE;

		if (up_idx < ur->start) {
			*pprev = prev;
			*pnext = ur;
			return TRUE;
		}
		prev = ur;
	}
	*pnext = NULL;
	*pprev = prev;

	if (prev && prev->start <= up_idx && up_idx < prev->start + prev->len)
		return FALSE;

	return TRUE;
}

static void
apply_mAOT_uprange(unsigned char epoch, unsigned idx_sm, unsigned idx_MTB, unsigned idx, skrid_t skrid, unsigned short offset)
{
	htod_copyinfo_t	*cinfo;
	unsigned	up_idx;
	uprange_t	*ur_new;
	uprange_t	*prev, *next;
	mAO_t	*mAO;

	pthread_spin_lock(&lock);
	up_idx = n_max_mtbs * epoch + mTB_INDEX_HOST(idx_sm, idx_MTB, idx) - 1;

	cinfo = copyinfos + COPYIDX_OTHER();

	mAO = mAO_EPOCH_HOST(cinfo, epoch, idx_sm, idx_MTB, idx);
	mAO->skrid = skrid;
	mAO->offset = offset;

	if (!get_sibling_upranges(cinfo, up_idx, &prev, &next)) {
		pthread_spin_unlock(&lock);
		return;
	}

	if (prev) {
		if (prev->start + prev->len == up_idx)
			prev->len++;
		else
			prev = NULL;
	}
	if (prev) {
		if (next) {
			if (prev->start + prev->len == next->start) {
				prev->len += next->len;
				list_del(&next->list);
				free(next);
			}
		}
		pthread_spin_unlock(&lock);
		return;
	}
	if (next) {
		if (up_idx == next->start - 1) {
			next->start = up_idx;
			next->len++;
			pthread_spin_unlock(&lock);
			return;
		}
	}

	ur_new = create_uprange(up_idx);
	if (next)
		list_add_tail(&ur_new->list, &next->list);
	else
		list_add_tail(&ur_new->list, &cinfo->upranges);

	pthread_spin_unlock(&lock);
}

static void
set_mtbs_skrid(sched_ctx_t *pctx, unsigned idx_sm, unsigned idx_MTB, unsigned n_mtbs, unsigned char *epochs)
{
	unsigned	n_mtbs_cur = 0;
	unsigned short	offbase = 0;
	unsigned	i;

	for (i = 1; i <= n_mtbs_per_MTB; i++) {
		unsigned char	epoch = epochs[i - 1];

		if (epoch == EPOCH_MAX)
			continue;

		apply_mAOT_uprange(epoch, idx_sm, idx_MTB, i, pctx->skrid, offbase);
		apply_mAOT_uprange(NEXT_EPOCH(epoch), idx_sm, idx_MTB, i, 0, (unsigned short)-1);
		EPOCH_HOST_ALLOC(idx_sm, idx_MTB, i) = NEXT_EPOCH(epoch);
		n_mtbs_cur++;
		if (n_mtbs_cur == n_mtbs)
			return;
		offbase++;
	}
}

static BOOL
assign_tb_by_rr(sched_ctx_t *pctx, unsigned n_mtbs)
{
	unsigned	i;

	for (i = 0; i < n_sm_count; i++) {
		unsigned	idx_sm, idx_MTB;

		lock_sm(&idx_sm, &idx_MTB);

		memset(pctx->epochs, EPOCH_MAX, n_mtbs_per_MTB);
		if (find_mtbs_on_sm(idx_sm, idx_MTB, n_mtbs, pctx->epochs)) {
			set_mtbs_skrid(pctx, idx_sm, idx_MTB, n_mtbs, pctx->epochs);
			unlock_sm(idx_sm, idx_MTB);

			return TRUE;
		}
		unlock_sm(idx_sm, idx_MTB);
	}

	return FALSE;
}

static void
reload_epochs(void)
{
	cuMemcpyDtoHAsync(mtb_epochs_host, (CUdeviceptr)g_mtb_epochs, n_max_mtbs, strm_host);
	cuStreamSynchronize(strm_host);
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
	pctx->epochs = (unsigned char *)malloc(n_mtbs_per_MTB);
}

static void
fini_sched_ctx(sched_ctx_t *pctx)
{
	free(pctx->epochs);
}

static void
schedule_mtbs(skrid_t skrid, unsigned n_mtbs_per_tb)
{
	sched_ctx_t	ctx;

	init_sched_ctx(&ctx, skrid);
	assign_tb_host(&ctx, n_mtbs_per_tb);
	fini_sched_ctx(&ctx);
}

static sk_t
submit_skrun_host(vstream_t vstream, skrun_t *skr)
{
	skrid_t	skrid;
	htod_copyinfo_t	*cinfo;

	pthread_spin_lock(&lock);

	while (skrid_done_min == (cur_skrid_host + 1) % n_queued_kernels) {
		/* full */
		pthread_spin_unlock(&lock);
		usleep(1000);
		pthread_spin_lock(&lock);
	}

	skrid = cur_skrid_host + 1;

	cinfo = copyinfos + COPYIDX_OTHER();
	if (cinfo->n_skruns == 0)
		cinfo->skrid_start = skrid;

	memcpy(cinfo->skruns + cinfo->n_skruns, skr, sizeof(skrun_t));
	cinfo->n_skruns++;
	cur_skrid_host = (cur_skrid_host + 1) % n_queued_kernels;

	pthread_spin_unlock(&lock);

	////TODO
	skrun_dones[skrid - 1] = FALSE;

	schedule_mtbs(skrid, skr->n_mtbs_per_tb);

	return (sk_t)(long long)skrid;
}

static void
wait_skrun_host(sk_t sk, vstream_t vstream, int *pres)
{
	skrun_t	*skr;
	skrid_t	skrid = (skrid_t)(long long)sk;

	pthread_mutex_lock(&mutex);

	while (!checker_done && !skrun_dones[skrid - 1])
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);

	skr = g_skruns + (skrid - 1);
	cuMemcpyDtoHAsync(pres, (CUdeviceptr)&skr->res, sizeof(int), strm_host);
	cuStreamSynchronize(strm_host);
}

static void
run_copycat(CUstream strm)
{
	htod_copyinfo_t	*cinfo;
	BOOL	sync_required = FALSE;
	struct list_head	*lp, *next;

	cinfo = copyinfos + copyidx;

	if (cinfo->n_skruns != 0) {
		CUresult res = cuMemcpyHtoDAsync((CUdeviceptr)(g_skruns + cinfo->skrid_start - 1), cinfo->skruns, sizeof(skrun_t) * cinfo->n_skruns, strm);
		sync_required = TRUE;
		cinfo->n_skruns = 0;
	}

	list_for_each_n (lp, &cinfo->upranges, next) {
		uprange_t	*ur = list_entry(lp, uprange_t, list);

		cuMemcpyHtoDAsync((CUdeviceptr)(g_mAOTs + ur->start), cinfo->mAOTs_host + ur->start, ur->len * sizeof(mAO_t), strm);
		sync_required = TRUE;
		list_del(&ur->list);
		free(ur);
	}

	if (sync_required)
		cuStreamSynchronize(strm);

	pthread_spin_lock(&lock);
	copyidx = COPYIDX_OTHER();
	pthread_spin_unlock(&lock);
}

static void *
htod_copycat_func(void *arg)
{
	cuCtxSetCurrent(context);

	while (!host_scheduler_done) {
		run_copycat(strm_host);
		usleep(10);
	}

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
init_skrun_host(void)
{
	void	*params[4];
	int	i;

	cuStreamCreate(&strm_host, CU_STREAM_NON_BLOCKING);

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
	params[2] = &g_skruns;
	params[3] = &g_mtbs_done;
	if (!invoke_kernel_func("func_init_skrun_host", params)) {
		exit(12);
	}

	mtb_epochs_host = (unsigned char *)malloc(n_max_mtbs);
	mtb_epochs_host_alloc = (unsigned char *)malloc(n_max_mtbs);
	for (i = 0; i < n_max_mtbs; i++) {
		mtb_epochs_host[i] = 0;
		mtb_epochs_host_alloc[i] = 0;
	}

	for (i = 0; i < 2; i++) {
		copyinfos[i].n_skruns = 0;
		copyinfos[i].skruns = (skrun_t *)malloc(sizeof(skrun_t) * n_queued_kernels);
		copyinfos[i].mAOTs_host = (mAO_t *)malloc(EPOCH_MAX * n_max_mtbs * sizeof(mAO_t));
		INIT_LIST_HEAD(&copyinfos[i].upranges);

		memset(copyinfos[i].mAOTs_host, 0, sizeof(mAO_t) * n_max_mtbs * EPOCH_MAX);
	}
	pthread_spin_init(&lock, 0);
	sched_sms = (sched_sm_t *)calloc(n_sm_count * n_MTBs_per_sm, sizeof(sched_sm_t));

	pthread_create(&host_scheduler, NULL, htod_copycat_func, NULL);
}

static void
fini_skrun_host(void)
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

sched_t	sched_sd_host = {
	"host",
	TBS_TYPE_SD_HOST,
	"func_macro_TB_host",
	init_skrun_host,
	fini_skrun_host,
	submit_skrun_host,
	wait_skrun_host,
};
