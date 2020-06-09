#include "mtbs_cu.h"

BOOL
run_native_tbs(unsigned *pticks)
{
	start_benchruns();

	init_tickcount();

	wait_benchruns();

	*pticks = get_tickcount();

	return TRUE;
}
