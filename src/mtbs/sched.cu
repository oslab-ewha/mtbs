#include "mtbs_cu.h"

#include "sched_cu.h"

#include <pthread.h>

static pthread_t	host_scheduler;
static BOOL	host_scheduler_done;

static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;

static sched_t	sched_hw = {
	"hw",
	TBS_TYPE_HW,
};

static sched_t	sched_sd_dynamic = {
	"dynamic",
	TBS_TYPE_SD_DYNAMIC,
};

static sched_t	sched_sd_static = {
	"static",
	TBS_TYPE_SD_STATIC,
};

static sched_t	*all_sched[] = {
	&sched_hw, &sched_sd_dynamic, &sched_sd_static, NULL
};

sched_t	*sched = &sched_hw;
unsigned	sched_id = 1;
char		*sched_argstr;

unsigned	n_max_mtbs;
unsigned	n_max_mtbs_per_sm;

static unsigned	mAT_uprange_start, mAT_uprange_end;

typedef struct {
	skrid_t		skrid;
	unsigned char	*epochs;
} sched_ctx_t;

#define NEXT_EPOCH(epoch)		(((epoch) + 1)  % EPOCH_MAX)
#define mTB_INDEX_HOST(id_sm, idx)	((id_sm - 1) * n_max_mtbs_per_sm + idx)
#define EPOCH_HOST(id_sm, idx)		mtb_epochs_host[mTB_INDEX_HOST(id_sm, idx) - 1]
#define EPOCH_HOST_ALLOC(id_sm, idx)	mtb_epochs_host_alloc[mTB_INDEX_HOST(id_sm, idx) - 1]
#define mTB_ALLOC_TABLE_EPOCH_HOST(epoch)	(mATs_host + n_max_mtbs * (epoch))
#define SKRID_EPOCH_HOST(epoch, id_sm, idx)	mTB_ALLOC_TABLE_EPOCH_HOST(epoch)[mTB_INDEX_HOST(id_sm, idx) - 1]

#define SET_ID_SM_NEXT(id_sm)	do { (id_sm) = (id_sm + 1) % n_sm_count; \
		if ((id_sm) == 0) (id_sm) = n_sm_count; } while (0)

static unsigned		id_sm_last = 1;

static CUstream		strm_sched;

static unsigned short	*g_mATs;
static unsigned	char	*g_mtb_epochs;
#if 0 ///TODO
static unsigned		epoch_cur;
#endif

static unsigned short	*mATs_host;
static unsigned	char	*mtb_epochs_host;
static unsigned	char	*mtb_epochs_host_alloc;

extern BOOL run_native_tbs(unsigned *pticks);
extern BOOL run_sd_tbs(unsigned *pticks);

extern void assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char skrid);

extern BOOL init_cuda(void);
extern void init_mem(void);
extern void init_skrun(void);
extern void fini_skrun(void);
extern void init_streams(void);
extern void fini_mem(void);

extern "C" void
setup_sched(const char *strpol)
{
	unsigned	i;

	for (i = 0; all_sched[i]; i++) {
		int	len = strlen(all_sched[i]->name);

		if (strncmp(strpol, all_sched[i]->name, len) == 0) {
			tbs_type_t	type;

			sched = all_sched[i];
			type = sched->type;
			sched_id = i + 1;

			if (strpol[len] ==':')
				sched_argstr = strdup(strpol + len + 1);
			else if (strpol[len] != '\0')
				continue;

			sched->name = strdup(strpol);
			sched->type = type;
			return;
		}
	}

	FATAL(1, "unknown scheduling policy: %s", strpol);
}

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

	pthread_mutex_lock(&mutex);
	up_idx = n_max_mtbs * epoch + mTB_INDEX_HOST(id_sm, idx) - 1;

	if (up_idx < mAT_uprange_start)
		mAT_uprange_start = up_idx;
	else if (up_idx >= mAT_uprange_end)
		mAT_uprange_end = up_idx + 1;
	pthread_mutex_unlock(&mutex);
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
	unsigned	id_sm_start = id_sm_last;
	int	id_sm_cur = id_sm_last;

	do {
		memset(pctx->epochs, EPOCH_MAX, n_max_mtbs_per_sm);
		if (find_mtbs_on_sm(id_sm_cur, n_mtbs, pctx->epochs)) {
			set_mtbs_skrid(pctx, id_sm_cur, n_mtbs, pctx->epochs);
			SET_ID_SM_NEXT(id_sm_cur);
			id_sm_last = id_sm_cur;

			return TRUE;
		}
		SET_ID_SM_NEXT(id_sm_cur);
	}
	while (id_sm_start != id_sm_cur);

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

void
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

static void
update_mAT(CUstream strm)
{
	unsigned	len;

	if (mAT_uprange_start == 0 && mAT_uprange_end == 0)
		return;

	len = mAT_uprange_end - mAT_uprange_start;

	cuMemcpyHtoDAsync((CUdeviceptr)(g_mATs + mAT_uprange_start), mATs_host + mAT_uprange_start, len * sizeof(unsigned short), strm);
	cuStreamSynchronize(strm);
	mAT_uprange_start = 0;
	mAT_uprange_end = 0;
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

extern "C" __global__ void
func_init_sched(int n_max_mtbs, unsigned short *g_mATs, unsigned char *g_mtb_epochs)
{
	extern __device__ unsigned short	*mATs;
	extern __device__ unsigned char		*mtb_epochs;
	int	i;

	mATs = g_mATs;
	mtb_epochs = g_mtb_epochs;

	for (i = 0; i < n_max_mtbs * EPOCH_MAX; i++) {
		mATs[i] = 0;
	}
	for (i = 0; i < n_max_mtbs; i++) {
		mtb_epochs[i] = 0;
	}
}

void
init_sched(void)
{
	CUfunction	func_init_sched;
	void		*params[3];
	CUresult	err;
	int	i;

	n_max_mtbs_per_sm = n_threads_per_MTB / N_THREADS_PER_mTB * n_MTBs_per_sm;
	n_max_mtbs = n_sm_count * n_max_mtbs_per_sm;

	g_mATs = (unsigned short *)mtbs_cudaMalloc(EPOCH_MAX * n_max_mtbs * sizeof(unsigned short));
	g_mtb_epochs = (unsigned char *)mtbs_cudaMalloc(n_max_mtbs);

	cuStreamCreate(&strm_sched, CU_STREAM_NON_BLOCKING);

	cuModuleGetFunction(&func_init_sched, mod, "func_init_sched");
	params[0] = &n_max_mtbs;
	params[1] = &g_mATs;
	params[2] = &g_mtb_epochs;

	err = cuLaunchKernel(func_init_sched, 1, 1, 1, 1, 1, 1, 0, strm_sched, params, NULL);
	if (err != CUDA_SUCCESS) {
		error("failed to initialize sched: %s\n", get_cuda_error_msg(err));
		exit(12);
	}
	cuStreamSynchronize(strm_sched);

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

	pthread_create(&host_scheduler, NULL, host_schedfunc, NULL);
}

void
fini_sched(void)
{
	void	*retval;

	host_scheduler_done = TRUE;
	pthread_join(host_scheduler, &retval);

	mtbs_cudaFree(g_mATs);
	mtbs_cudaFree(g_mtb_epochs);
}

extern "C" BOOL
run_tbs(unsigned *pticks)
{
	BOOL	res;

	if (!init_cuda())
		return FALSE;

	init_mem();
	init_skrun();
	init_benchruns();
	init_streams();

	if (sched->type == TBS_TYPE_HW)
		res = run_native_tbs(pticks);
	else
		res = run_sd_tbs(pticks);

	fini_skrun();
	fini_mem();

	return res;
}
