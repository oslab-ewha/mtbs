#include "mtbs_cu.h"

#include <pthread.h>

#include "sched_dyn.cuh"
#include "mAT.h"

static sk_t
submit_skrun_dyn(vstream_t vstream, skrun_t *skr)
{
	return (sk_t)(long long)submit_skrun_mAT(skr);
}

static void
wait_skrun_dyn(sk_t sk, vstream_t vstream, int *pres)
{
	return wait_skrun_mAT(sk, pres);
}

static void
init_skrun_dyn(void)
{
	init_skrun_mAT();
	invoke_kernel_func("setup_sched_dyn", NULL);
}

static void
fini_skrun_dyn(void)
{
	fini_skrun_mAT();
}

sched_t	sched_sd_dynamic = {
	"dynamic",
	TBS_TYPE_SD_DYNAMIC,
	"func_macro_TB_mAT",
	init_skrun_dyn,
	fini_skrun_dyn,
	submit_skrun_dyn,
	wait_skrun_dyn,
};
