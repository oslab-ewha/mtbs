#include "mtbs_cu.h"

CUmodule	mod;

extern "C" unsigned	arg_n_MTBs_per_sm;
extern "C" unsigned	arg_n_threads_per_MTB;

static BOOL
setup_gpu_devinfo(CUdevice dev)
{
	unsigned	max_threads_per_sm, max_threads_per_block;
	CUresult	err;

	err = cuDeviceGetAttribute((int *)&n_sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
	if (err != CUDA_SUCCESS) {
		error("failed to get gpu device property(sm count): %s", get_cuda_error_msg(err));
		return FALSE;
	}

	err = cuDeviceGetAttribute((int *)&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
	if (err != CUDA_SUCCESS) {
		error("failed to get gpu device property(max threads per block): %s", get_cuda_error_msg(err));
		return FALSE;
	}
	err = cuDeviceGetAttribute((int *)&max_threads_per_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
	if (err != CUDA_SUCCESS) {
		error("failed to get gpu device property(max threads per sm): %s", get_cuda_error_msg(err));
		return FALSE;
	}

	if (arg_n_MTBs_per_sm == 0 && arg_n_threads_per_MTB == 0) {
		n_threads_per_MTB = max_threads_per_block;
		n_MTBs_per_sm = max_threads_per_sm / n_threads_per_MTB;
	}
	else if (arg_n_MTBs_per_sm > 0) {
		n_MTBs_per_sm = arg_n_MTBs_per_sm;
		if (arg_n_threads_per_MTB > 0)
			n_threads_per_MTB = arg_n_threads_per_MTB;
		else
			n_threads_per_MTB = max_threads_per_sm / n_MTBs_per_sm;
	}
	else {
		n_threads_per_MTB = arg_n_threads_per_MTB;
		n_MTBs_per_sm = max_threads_per_sm / n_threads_per_MTB;
	}

	if (n_threads_per_MTB > max_threads_per_block)
		n_threads_per_MTB = max_threads_per_block;
	if (n_threads_per_MTB < 32) {
		error("Too small threads per MTB: %d", n_threads_per_MTB);
		return FALSE;
	}
	if (n_threads_per_MTB % 32) {
		error("Invalid thread count per MTB: %d", n_threads_per_MTB);
		return FALSE;
	}

	return TRUE;
}

BOOL
init_cuda(void)
{
	CUresult	res;
	CUdevice	dev;

	cuInit(0);
	res = cuDeviceGet(&dev, devno);
	if (res != CUDA_SUCCESS) {
		error("failed to get device: %s", get_cuda_error_msg(res));
		return FALSE;
	}

	if (!setup_gpu_devinfo(dev)) {
		return FALSE;
	}

	res = cuDevicePrimaryCtxRetain(&context, dev);
	if (res != CUDA_SUCCESS) {
		error("failed to get context: %s", get_cuda_error_msg(res));
		return FALSE;
	}

	res = cuCtxSetCurrent(context);
	if (res != CUDA_SUCCESS) {
		error("failed to set context: %s", get_cuda_error_msg(res));
		return FALSE;
	}

	res = cuModuleLoad(&mod, "mtbs.cubin");
	if (res != CUDA_SUCCESS) {
		error("failed to load module mtbs.cubin: %s", get_cuda_error_msg(res));
		return FALSE;
	}

	return TRUE;
}
