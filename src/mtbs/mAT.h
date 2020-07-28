#ifndef _MAT_H_
#define _MAT_H_

#include "mtbs_cu.h"

#define EPOCH_MAX		64

#define mTB_TOTAL_COUNT()	(d_fkinfo->n_max_mtbs_per_sm * d_fkinfo->n_sm_count)

#define mTB_INDEX(id_sm, idx)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + idx)
#define mTB_INDEX_MY(id_sm)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)

#define EPOCH(id_sm, idx)	mtb_epochs[mTB_INDEX(id_sm, idx) - 1]
#define EPOCH_MY(id_sm)		mtb_epochs[mTB_INDEX_MY(id_sm) - 1]

#define IS_LEADER_THREAD()	(threadIdx.x % N_THREADS_PER_mTB == 0)

#define mTB_ALLOC_TABLE_EPOCH(epch)	(mATs + mTB_TOTAL_COUNT() * (epch))
#define mTB_ALLOC_TABLE(id_sm, idx)	(mATs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_ALLOC_TABLE_MY(id_sm)	(mATs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))
#define SKRID(id_sm, idx)	mTB_ALLOC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define SKRID_MY(id_sm)		mTB_ALLOC_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define SKR_N_TBS_SCHED(skrid)	skr_n_tbs_sched[skrid - 1]

#define BRK_INDEX_EPOCH(id_sm, idx, epch)	mTB_ALLOC_TABLE_EPOCH(epch)[mTB_INDEX(id_sm, idx) - 1]

#define BRK_N_MTBS_ASSIGNABLE(brid)	brk_n_mtbs_assignable[brid - 1]

#define mTB_OFFSET_TABLE_EPOCH(epch)	(mOTs + mTB_TOTAL_COUNT() * (epch))
#define mTB_OFFSET_TABLE(id_sm, idx)	(mOTs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_OFFSET_TABLE_MY(id_sm)	(mOTs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))

#define mTB_OFFSET_TB_EPOCH(id_sm, idx, epch)	mTB_OFFSET_TABLE_EPOCH(epch)[mTB_INDEX(id_sm, idx) - 1]
#define mTB_OFFSET_TB(id_sm, idx)	mTB_OFFSET_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define mTB_OFFSET_TB_MY(id_sm)		mTB_OFFSET_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define mTB_SYNC_TABLE_EPOCH(epch)    (mSTs + mTB_TOTAL_COUNT() * (epch))
#define mTB_SYNC_TABLE(id_sm, idx)    (mSTs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))

#define mTB_SYNC_EPOCH(id_sm, idx, epch)	mTB_SYNC_TABLE_EPOCH(epch)[mTB_INDEX(id_sm, idx) - 1]
#define mTB_SYNC(id_sm, idx)	mTB_SYNC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]

extern __device__ volatile unsigned short	*mATs;
extern __device__ volatile unsigned char	*mtb_epochs;
extern __device__ volatile unsigned short	*mOTs;
extern __device__ volatile unsigned short	*mSTs;

extern void __device__ (*run_schedule)(void);

extern void init_skrun_mAT(void);
extern void fini_skrun_mAT(void);

extern skrid_t submit_skrun_mAT(skrun_t *skr);
extern void wait_skrun_mAT(sk_t sk, int *pres);

#endif
