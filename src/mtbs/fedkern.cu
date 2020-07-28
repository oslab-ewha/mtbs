#include "mtbs_cu.h"

#include "tbs_sd.h"

fedkern_info_t	*g_fkinfo;

__device__ fedkern_info_t	*d_fkinfo;

extern unsigned	n_max_mtbs_per_sm;

extern "C" __global__ void
func_setup_fedkern(fedkern_info_t *fkinfo)
{
	d_fkinfo = fkinfo;
}

void
create_fedkern_info(void)
{
	fedkern_info_t	*fkinfo;
	void	*params[1];

	fkinfo = (fedkern_info_t *)calloc(1, sizeof(fedkern_info_t));

	fkinfo->n_sm_count = n_sm_count;
	fkinfo->sched_id = sched_id;
	fkinfo->n_mtbs = n_mtbs_submitted;
	fkinfo->n_max_mtbs_per_sm = n_max_mtbs_per_sm;
	fkinfo->n_max_mtbs_per_MTB = n_max_mtbs_per_sm / n_MTBs_per_sm;
	fkinfo->sched_done = FALSE;

	g_fkinfo = (fedkern_info_t *)mtbs_cudaMalloc(sizeof(fedkern_info_t));
	cuMemcpyHtoD((CUdeviceptr)g_fkinfo, fkinfo, sizeof(fedkern_info_t));

	params[0] = &g_fkinfo;
	invoke_kernel_func("func_setup_fedkern", params);

	free(fkinfo);
}

void
free_fedkern_info(void)
{
	mtbs_cudaFree(g_fkinfo);
}
