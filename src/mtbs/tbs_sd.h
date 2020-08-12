#ifndef _TBS_SD_H_
#define _TBS_SD_H_

#include "../../config.h"

extern fedkern_info_t	*g_fkinfo;
extern __device__ fedkern_info_t	*d_fkinfo;

#define IS_LEADER_THREAD()	(threadIdx.x % N_THREADS_PER_mTB == 0)

#if CUDA_COMPUTE >= 60
#define SYNCWARP()	__syncwarp()
#else
#define SYNCWARP()	do {} while (0)
#endif

typedef struct {
	skrun_t *(*get_skr)(void);
	unsigned short (*get_offset_TB)(void);
	unsigned (*get_barid)(skrun_t *skr);
} benchapi_funcs_t;

extern __device__ benchapi_funcs_t	benchapi_funcs;

#endif
