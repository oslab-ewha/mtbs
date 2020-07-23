#include "mtbs_cu.h"

extern unsigned	n_max_mtbs_per_sm;

fedkern_info_t *
create_fedkern_info(void)
{
	fedkern_info_t	*fkinfo;
	fedkern_info_t	*g_fkinfo;

	fkinfo = (fedkern_info_t *)calloc(1, sizeof(fedkern_info_t));

	fkinfo->n_sm_count = n_sm_count;
	fkinfo->sched_id = sched_id;
	fkinfo->n_mtbs = n_mtbs_submitted;
	fkinfo->n_max_mtbs_per_sm = n_max_mtbs_per_sm;
	fkinfo->n_max_mtbs_per_MTB = n_max_mtbs_per_sm / n_MTBs_per_sm;
	fkinfo->sched_done = FALSE;

	g_fkinfo = (fedkern_info_t *)mtbs_cudaMalloc(sizeof(fedkern_info_t));
	cuMemcpyHtoD((CUdeviceptr)g_fkinfo, fkinfo, sizeof(fedkern_info_t));

	return g_fkinfo;
}

void
free_fedkern_info(fedkern_info_t *g_fkinfo)
{
	mtbs_cudaFree(g_fkinfo);
}

void
wait_fedkern_initialized(fedkern_info_t *d_fkinfo)
{
	CUstream	strm;

	cuStreamCreate(&strm, CU_STREAM_NON_BLOCKING);

	while (TRUE) {
		BOOL	initialized = FALSE;

		cuMemcpyDtoHAsync(&initialized, (CUdeviceptr)&d_fkinfo->initialized, sizeof(BOOL), strm);
		cuStreamSynchronize(strm);
		if (initialized)
			break;
	}
	cuStreamDestroy(strm);
}
