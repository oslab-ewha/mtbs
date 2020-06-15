#ifndef _STREAM_H_
#define _STREAM_H_

typedef struct {
	cudaStream_t	cudaStrm;
	unsigned	refcnt;
} vStrm_t, *vstrm_t;

#endif
