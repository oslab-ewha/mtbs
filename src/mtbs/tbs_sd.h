#ifndef _TBS_SD_H_
#define _TBS_SD_H_

extern fedkern_info_t	*g_fkinfo;
extern __device__ fedkern_info_t	*d_fkinfo;

#if CUDA_COMPUTE >= 60
#define SYNCWARP()	__syncwarp()
#else
#define SYNCWARP()	do {} while (0)
#endif

#endif
