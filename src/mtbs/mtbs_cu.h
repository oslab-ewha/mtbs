#ifndef _SDTBS_CU_H_
#define _SDTBS_CU_H_

#include "mtbs.h"

#include "../../benchmarks/benchapi.h"

#include <cuda.h>

#define MAX_QUEUED_KERNELS	512
#define MAX_ARGS	5
#define N_THREADS_PER_mTB	32

extern unsigned n_sm_count;
extern unsigned n_threads_per_MTB;
extern unsigned	n_MTBs_per_sm;
extern int	n_benches;
extern int	n_tbs_submitted;
extern int	n_mtbs_submitted;

extern unsigned		n_max_mtbs;
extern unsigned		n_max_mtbs_per_sm;

extern unsigned		n_queued_kernels;

extern CUcontext	context;
extern CUmodule		mod;

typedef unsigned short	skrid_t;

typedef int (*cookarg_func_t)(dim3 dimGrid, dim3 dimBlock, void *args[]);
typedef int (*bench_func_t)(dim3 dimGrid, dim3 dimBlock, void *args[]);

typedef struct {
	skid_t		skid;
	void		*args[MAX_ARGS];
	int		res;
	dim3		dimGrid, dimBlock;
	unsigned	n_tbs, n_mtbs_per_tb;
} skrun_t;

typedef enum {
	TBS_TYPE_HW = 1,
	TBS_TYPE_SD_DYNAMIC,
	TBS_TYPE_SD_STATIC,
	TBS_TYPE_SD_PAGODA,
	TBS_TYPE_SD_GEMTC,
} tbs_type_t;

typedef struct {
	BOOL		sched_done, going_to_shutdown;
	tbs_type_t	tbs_type;
	unsigned	n_sm_count;
	unsigned	n_max_mtbs_per_sm;
	unsigned	n_max_mtbs_per_MTB;
	unsigned	n_mtbs;
	unsigned	n_queued_kernels;
} fedkern_info_t;

typedef struct {
	const char	*code;
	skid_t		skid;
	cookarg_func_t	cookarg_func;
	bench_func_t	bench_func;
} benchinfo_t;

typedef struct {
	benchinfo_t	*info;
	dim3	dimGrid, dimBlock;
	void	*args[MAX_ARGS];
	int	res;
} benchrun_t;

typedef struct {
	const char	*name;
	tbs_type_t	type;
	const char	*macro_TB_funcname;
	void (*init_skrun)(void);
	void (*fini_skrun)(void);
	sk_t (*submit_skrun)(vstream_t vstream, skrun_t *skr);
	void (*wait_skrun)(sk_t sk, vstream_t vstream, int *pres);
} sched_t;

__device__ extern skrun_t	*d_skruns;

extern sched_t		*sched;
extern benchrun_t	*benchruns;

extern void init_benchruns(void);
extern void start_benchruns(void);
extern void wait_benchruns(void);

__device__ uint get_smid(void);
__device__ uint get_laneid(void);
__device__ void sleep_in_kernel(void);
__device__ unsigned find_mtb_start(unsigned id_sm, unsigned idx_mtb_start, unsigned n_mtbs);
__device__ unsigned get_n_active_mtbs(unsigned id_sm);

BOOL invoke_kernel_func(const char *funcname, void **params);

unsigned long long get_ticks(void);

__device__ void run_sub_kernel(skrun_t *skr);

void create_fedkern_info(void);
void free_fedkern_info(void);

unsigned get_n_mTBs_for_threads(unsigned n_threads);
BOOL is_sm_avail(int id_sm, unsigned n_mTBs);
unsigned get_sm_n_sched_mTBs(int id_sm);
void use_next_mAT(int id_sm);

void init_tickcount(void);
unsigned get_tickcount(void);

const char *get_cuda_error_msg(CUresult err);

#endif
