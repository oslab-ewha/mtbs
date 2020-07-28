#include "mtbs_cu.h"

#include "stream.h"

extern CUfunction	func_sub_kernel;

static sk_t
submit_skrun_hw(vstream_t stream, skrun_t *skr)
{
	skrun_t	*d_skr;
	CUstream	cstrm = NULL;
	void	*params[1];
	CUresult	err;

	if (stream != NULL) {
		cstrm = ((vstrm_t)stream)->cudaStrm;
	}

	d_skr = (skrun_t *)mtbs_cudaMalloc(sizeof(skrun_t));
	cuMemcpyHtoDAsync((CUdeviceptr)d_skr, skr, sizeof(skrun_t), cstrm);

	params[0] = &d_skr;
	err = cuLaunchKernel(func_sub_kernel,
			     skr->dimGrid.x, skr->dimGrid.y, skr->dimGrid.z,
			     skr->dimBlock.x, skr->dimBlock.y, skr->dimBlock.z,
			     0, cstrm, params, NULL);
	if (err != CUDA_SUCCESS) {
		error("hw kernel error: %s", get_cuda_error_msg(err));
	}
	return d_skr;
}

static void
wait_skrun_hw(sk_t sk, vstream_t vstream, int *pres)
{
	CUstream	cstrm;
	skrun_t	*d_skr = (skrun_t *)sk;

	cstrm = ((vstrm_t)vstream)->cudaStrm;

	cuStreamSynchronize(cstrm);
	cuMemcpyDtoHAsync(pres, (CUdeviceptr)&d_skr->res, sizeof(int), cstrm);
	cuStreamSynchronize(cstrm);

	mtbs_cudaFree(d_skr);
}

sched_t	sched_hw = {
	"hw",
	TBS_TYPE_HW,
	NULL,
	NULL,
	NULL,
	submit_skrun_hw,
	wait_skrun_hw
};
