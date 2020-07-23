#ifndef _STREAM_H_
#define _STREAM_H_

#include <cuda.h>

typedef struct {
	CUstream	cudaStrm;
	unsigned	refcnt;
} vStrm_t, *vstrm_t;

#endif
