#ifndef _BENCHAPI_H_
#define _BENCHAPI_H_

#define LOOPCALC	((skid_t)1)
#define MKLC		((skid_t)2)
#define GMA		((skid_t)3)
#define LMA		((skid_t)4)
#define KMEANS		((skid_t)5)
#define MANDELBROT	((skid_t)6)
#define IRREGULAR	((skid_t)7)
#define MM		((skid_t)8)

typedef unsigned char	skid_t;
typedef void *	sk_t;
typedef void *	vstream_t;

__device__ unsigned get_random(unsigned randx);

__device__ int get_gridDimX(void);
__device__ int get_gridDimY(void);
__device__ int get_blockIdxX(void);
__device__ int get_blockIdxY(void);
__device__ int get_blockDimX(void);
__device__ int get_blockDimY(void);
__device__ int get_threadIdxX(void);
__device__ int get_threadIdxY(void);

__device__ void sync_threads(void);

sk_t launch_kernel(skid_t skid, vstream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[]);
void wait_kernel(sk_t sk, vstream_t strm, int *res);

vstream_t create_vstream(void);
void destroy_vstream(vstream_t vstream);

void *mtbs_cudaMalloc(unsigned size);
void mtbs_cudaFree(void *ptr);

#endif
