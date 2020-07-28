#include "mtbs_cu.h"

extern unsigned	verbose;

extern "C" void
report(unsigned elapsed)
{
	benchrun_t	*brun;

	printf("tbs type: %s\n", sched->name);
	if (sched->type != TBS_TYPE_HW) {
		printf("sm count: %u\n", n_sm_count);
		printf("n threads per MTB: %u\n", n_threads_per_MTB);
		printf("n MTBs per SM: %u\n", n_MTBs_per_sm);
	}
	printf("elapsed time: %.6lf\n", elapsed / 1000000.0);
	brun = benchruns;
	if (verbose > 1) {
		int	i;
		for (i = 0; i < n_benches; i++, brun++) {
			printf("%s: %d\n", brun->info->code, brun->res);
		}
	}
}
